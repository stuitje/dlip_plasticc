from __future__ import annotations

import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

from avocado.classifier import Classifier
from avocado.settings import settings
from avocado.utils import AvocadoException

# Optional PyTorch import at module level.
_TORCH_IMPORT_ERROR = None
try:
    import torch
    from torch import nn
except Exception as exc:
    torch = None
    _TORCH_IMPORT_ERROR = exc

    class _FallbackModule:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for TransformerClassifier but is not installed."
            ) from _TORCH_IMPORT_ERROR

    class _NNFallback:
        Module = _FallbackModule

    nn = _NNFallback()


class _LearnedTimeEmbedding(nn.Module):
    def __init__(self, d_model, hidden_dim=32, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, times, cont_x):
        dt = torch.diff(times, dim=1, prepend=times[:, :1])
        dt = torch.clamp(dt, min=0.0)

        if cont_x.shape[-1] > 1:
            log_dt = cont_x[:, :, 1]
        else:
            log_dt = torch.log1p(dt)

        time_feats = torch.stack([times, dt, log_dt], dim=-1)
        emb = self.net(time_feats)
        return self.norm(emb)


class _AttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        scores = self.score(x).squeeze(-1)

        if mask is None:
            weights = torch.softmax(scores, dim=1)
            pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
            return pooled

        mask = mask.to(dtype=x.dtype)
        neg_large = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask == 0, neg_large)

        weights = torch.softmax(scores, dim=1) * mask
        weight_sums = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weights = weights / weight_sums

        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled


class _TransformerNet(nn.Module):
    def __init__(
        self,
        cont_dim,
        num_classes,
        num_bands=6,
        global_dim=5,
        d_model=128,
        band_emb_dim=16,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
        time_hidden_dim=32,
    ):
        super().__init__()

        if band_emb_dim >= d_model:
            raise ValueError("band_emb_dim must be smaller than d_model.")

        cont_proj_dim = d_model - band_emb_dim

        self.cont_norm = nn.LayerNorm(cont_dim)
        self.cont_proj = nn.Linear(cont_dim, cont_proj_dim)
        self.band_emb = nn.Embedding(num_bands, band_emb_dim)
        self.input_norm = nn.LayerNorm(d_model)

        self.time_emb = _LearnedTimeEmbedding(
            d_model=d_model,
            hidden_dim=time_hidden_dim,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = _AttentionPool(d_model)

        self.global_mlp = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, 64),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model + 64),
            nn.Linear(d_model + 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, cont_x, band_ids, mask=None, global_feats=None):
        times = cont_x[:, :, 0]

        cont_x_norm = self.cont_norm(cont_x)
        x_cont = self.cont_proj(cont_x_norm)

        band_ids = torch.clamp(band_ids, min=0, max=self.band_emb.num_embeddings - 1)
        x_band = self.band_emb(band_ids)

        x = torch.cat([x_cont, x_band], dim=-1)
        x = self.input_norm(x)
        x = x + self.time_emb(times, cont_x)

        pad_mask = None
        if mask is not None:
            pad_mask = mask == 0

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        pooled = self.pool(x, mask)

        if global_feats is None:
            raise ValueError("global_feats must be provided.")

        global_repr = self.global_mlp(global_feats)
        fused = torch.cat([pooled, global_repr], dim=-1)

        return self.head(fused)


class TransformerClassifier(Classifier):
    def __init__(
        self,
        name,
        featurizer,
        num_epochs=15,
        batch_size=64,
        lr=1e-4,
        class_weights=None,
        auto_class_weights=False,
        device=None,
        d_model=128,
        band_emb_dim=16,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
        num_bands=6,
        time_hidden_dim=32,
    ):
        super().__init__(name)

        self.featurizer = featurizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.class_weights = class_weights
        self.auto_class_weights = auto_class_weights
        self.device = device

        self.d_model = d_model
        self.band_emb_dim = band_emb_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_bands = num_bands
        self.time_hidden_dim = time_hidden_dim

        self.model = None
        self.class_names = None
        self.best_val_loss = None
        self.history = None
        self.val_fold = None

    def _ensure_torch(self):
        if _TORCH_IMPORT_ERROR is not None:
            raise AvocadoException(
                "PyTorch is required for TransformerClassifier but is not installed."
            ) from _TORCH_IMPORT_ERROR

    def _build_class_weight_tensor(self, class_names, class_indices_train, device):
        if not self.auto_class_weights and self.class_weights is None:
            return None

        weights = np.ones(len(class_names), dtype=np.float32)

        if self.auto_class_weights:
            counts = np.bincount(
                class_indices_train, minlength=len(class_names)
            ).astype(np.float32)
            counts = np.where(counts == 0, 1.0, counts)
            inv_freq = 1.0 / counts
            inv_freq /= inv_freq.mean()
            weights *= inv_freq

        if self.class_weights is not None:
            for i, c in enumerate(class_names):
                weights[i] *= self.class_weights.get(c, 1.0)

        return torch.tensor(weights, dtype=torch.float32).to(device)

    def _derive_global_features(self, cont_features, mask):
        valid = mask.astype(np.float32)
        denom = np.clip(valid.sum(axis=1), 1.0, None)

        flux = cont_features[:, :, 2]
        flux_err = cont_features[:, :, 3]
        snr = cont_features[:, :, 4]
        detected = cont_features[:, :, 5]

        valid_frac = valid.mean(axis=1)
        mean_abs_flux = (np.abs(flux) * valid).sum(axis=1) / denom
        mean_flux_err = (np.abs(flux_err) * valid).sum(axis=1) / denom
        mean_detected = (detected * valid).sum(axis=1) / denom

        masked_abs_snr = np.where(valid > 0, np.abs(snr), -np.inf)
        max_abs_snr = masked_abs_snr.max(axis=1)
        max_abs_snr[~np.isfinite(max_abs_snr)] = 0.0

        global_feats = np.stack(
            [valid_frac, mean_abs_flux, mean_flux_err, max_abs_snr, mean_detected],
            axis=1,
        ).astype(np.float32)

        return global_feats

    def _parse_features(self, dataset):
        features = dataset.select_features(self.featurizer)

        cont_features = None
        band_ids = None
        mask = None
        global_feats = None

        if isinstance(features, (tuple, list)):
            if len(features) == 4:
                cont_features, band_ids, mask, global_feats = features
            elif len(features) == 3:
                cont_features, band_ids, mask = features
            elif len(features) == 2:
                features, mask = features
            else:
                raise AvocadoException(
                    "Unsupported feature tuple format for TransformerClassifier."
                )

        if cont_features is None and band_ids is None:
            if hasattr(features, "values"):
                features = features.values
            features = np.asarray(features, dtype=float)

            if np.isnan(features).any() or np.isinf(features).any():
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            if features.ndim != 3:
                raise AvocadoException(
                    "TransformerClassifier expects features shaped "
                    "(n, seq_len, channels), (features, mask), "
                    "(cont_features, band_ids, mask), or "
                    "(cont_features, band_ids, mask, global_features)."
                )

            if features.shape[2] < 5:
                raise AvocadoException(
                    "Legacy sequence features must have at least 5 channels."
                )

            if mask is None:
                mask = np.ones(features.shape[:2], dtype=np.float32)
            else:
                if hasattr(mask, "values"):
                    mask = mask.values
                mask = np.asarray(mask, dtype=np.float32)

            time_chan = features[:, :, 0].astype(np.float32)
            flux_chan = features[:, :, 1].astype(np.float32)
            flux_err_chan = features[:, :, 2].astype(np.float32)
            detected_chan = features[:, :, 3].astype(np.float32)
            band_ids = np.rint(features[:, :, -1]).astype(np.int64)

            dt = np.diff(time_chan, axis=1, prepend=time_chan[:, :1])
            dt = np.clip(dt, 0.0, None)
            log_dt = np.log1p(dt).astype(np.float32)

            snr = flux_chan / (np.abs(flux_err_chan) + 1e-8)
            snr = np.clip(snr, -20.0, 20.0).astype(np.float32)

            extra_cont = None
            if features.shape[2] > 5:
                extra_cont = features[:, :, 4:-1].astype(np.float32)

            base_cont = [
                time_chan[:, :, None],
                log_dt[:, :, None],
                flux_chan[:, :, None],
                flux_err_chan[:, :, None],
                snr[:, :, None],
                detected_chan[:, :, None],
            ]

            if extra_cont is not None and extra_cont.shape[2] > 0:
                base_cont.append(extra_cont)

            cont_features = np.concatenate(base_cont, axis=2).astype(np.float32)

        else:
            if hasattr(cont_features, "values"):
                cont_features = cont_features.values
            if hasattr(band_ids, "values"):
                band_ids = band_ids.values
            if hasattr(mask, "values"):
                mask = mask.values
            if global_feats is not None and hasattr(global_feats, "values"):
                global_feats = global_feats.values

            cont_features = np.asarray(cont_features, dtype=float)
            band_ids = np.asarray(band_ids, dtype=np.int64)
            mask = np.asarray(mask, dtype=np.float32)

            if np.isnan(cont_features).any() or np.isinf(cont_features).any():
                cont_features = np.nan_to_num(
                    cont_features, nan=0.0, posinf=0.0, neginf=0.0
                )

            if cont_features.ndim != 3:
                raise AvocadoException("cont_features must have shape (n, seq_len, channels).")
            if band_ids.ndim != 2:
                raise AvocadoException("band_ids must have shape (n, seq_len).")
            if mask.ndim != 2:
                raise AvocadoException("mask must have shape (n, seq_len).")

            cont_features = cont_features.astype(np.float32)

        if global_feats is None:
            global_feats = self._derive_global_features(cont_features, mask)
        else:
            global_feats = np.asarray(global_feats, dtype=float)
            if np.isnan(global_feats).any() or np.isinf(global_feats).any():
                global_feats = np.nan_to_num(
                    global_feats, nan=0.0, posinf=0.0, neginf=0.0
                )
            global_feats = global_feats.astype(np.float32)

        return (
            cont_features.astype(np.float32),
            band_ids.astype(np.int64),
            mask.astype(np.float32),
            global_feats.astype(np.float32),
        )

    def train(
        self,
        dataset,
        num_folds=None,
        random_state=None,
        val_fold=0,
        show_progress=True,
        weight_decay=1e-2,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=2,
        min_lr=1e-6,
        early_stopping_patience=5,
    ):
        self._ensure_torch()

        try:
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as exc:
            raise AvocadoException(
                "PyTorch is required for TransformerClassifier but is not installed."
            ) from exc

        if num_folds is None:
            num_folds = settings["num_folds"]
        if random_state is None:
            random_state = settings["fold_random_state"]

        if num_folds < 2:
            raise AvocadoException("TransformerClassifier requires num_folds >= 2.")

        cont_features, band_ids, mask, global_feats = self._parse_features(dataset)

        object_classes = dataset.metadata["class"].values
        class_names = np.unique(object_classes)
        class_map = {c: i for i, c in enumerate(class_names)}
        class_indices = np.array([class_map[c] for c in object_classes], dtype=np.int64)

        folds = dataset.label_folds(num_folds, random_state)
        val_mask = folds == val_fold
        train_mask = folds != val_fold

        if np.sum(val_mask) == 0:
            raise AvocadoException("Validation fold is empty.")
        if np.sum(train_mask) == 0:
            raise AvocadoException("Training split is empty.")

        Xc_train = torch.tensor(cont_features[train_mask], dtype=torch.float32)
        Xb_train = torch.tensor(band_ids[train_mask], dtype=torch.long)
        M_train  = torch.tensor(mask[train_mask], dtype=torch.float32)
        G_train  = torch.tensor(global_feats[train_mask], dtype=torch.float32)
        y_train  = torch.tensor(class_indices[train_mask], dtype=torch.long)

        Xc_val = torch.tensor(cont_features[val_mask], dtype=torch.float32)
        Xb_val = torch.tensor(band_ids[val_mask], dtype=torch.long)
        M_val  = torch.tensor(mask[val_mask], dtype=torch.float32)
        G_val  = torch.tensor(global_feats[val_mask], dtype=torch.float32)
        y_val  = torch.tensor(class_indices[val_mask], dtype=torch.long)

        # ── FIX 1: pin_memory=False prevents CUDA pinned memory accumulation ──
        train_ds = TensorDataset(Xc_train, Xb_train, M_train, G_train, y_train)
        val_ds   = TensorDataset(Xc_val,   Xb_val,   M_val,   G_val,   y_val)

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            pin_memory=False, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            pin_memory=False, num_workers=0,
        )

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        num_bands_used = max(self.num_bands, int(np.max(band_ids)) + 1)

        model = _TransformerNet(
            cont_dim=cont_features.shape[2],
            num_classes=len(class_names),
            num_bands=num_bands_used,
            global_dim=global_feats.shape[1],
            d_model=self.d_model,
            band_emb_dim=self.band_emb_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            time_hidden_dim=self.time_hidden_dim,
        ).to(device)

        weight_tensor = self._build_class_weight_tensor(
            class_names, class_indices[train_mask], device
        )
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            threshold=1e-4,
            min_lr=min_lr,
        )

        iterator = range(self.num_epochs)
        if show_progress:
            iterator = tqdm(iterator, desc="Epochs", dynamic_ncols=True)

        best_val_loss = np.inf
        best_state = None
        history = []
        epochs_without_improvement = 0

        for epoch in iterator:
            # ---- train ----
            model.train()
            train_loss = 0.0

            for xc, xb, mb, gb, yb in train_loader:
                xc = xc.to(device, non_blocking=True)
                xb = xb.to(device, non_blocking=True)
                mb = mb.to(device, non_blocking=True)
                gb = gb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                logits = model(xc, xb, mb, gb)
                loss = loss_fn(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * xc.size(0)

                # ── FIX 2: explicitly delete batch tensors each step ──────────
                del xc, xb, mb, gb, yb, logits, loss

            train_loss /= len(train_ds)

            # ---- validation ----
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for xc, xb, mb, gb, yb in val_loader:
                    xc = xc.to(device, non_blocking=True)
                    xb = xb.to(device, non_blocking=True)
                    mb = mb.to(device, non_blocking=True)
                    gb = gb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                    logits = model(xc, xb, mb, gb)
                    loss = loss_fn(logits, yb)
                    val_loss += loss.item() * xc.size(0)

                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)

                    # ── FIX 2 (val): same cleanup ─────────────────────────────
                    del xc, xb, mb, gb, yb, logits, loss, preds

            val_loss /= len(val_ds)
            val_acc = correct / total if total > 0 else np.nan

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # ── FIX 3: clear GPU cache + run gc each epoch ────────────────────
            torch.cuda.empty_cache()
            gc.collect()

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc) if not np.isnan(val_acc) else np.nan,
                    "lr": float(current_lr),
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            msg = (
                "Epoch %d train_loss: %.5f val_loss: %.5f "
                "val_acc: %.5f lr: %.6e"
                % (epoch + 1, train_loss, val_loss, val_acc, current_lr)
            )
            if show_progress:
                tqdm.write(msg)
            else:
                print(msg)

            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                stop_msg = (
                    "Early stopping at epoch %d. Best val_loss: %.5f"
                    % (epoch + 1, best_val_loss)
                )
                if show_progress:
                    tqdm.write(stop_msg)
                else:
                    print(stop_msg)
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model = model
        self.class_names = class_names
        self.device = device
        self.best_val_loss = best_val_loss
        self.history = pd.DataFrame(history)
        self.val_fold = val_fold

        print("Best validation loss: %.5f" % best_val_loss)

        return model

    def predict(self, dataset, show_progress=True):
        self._ensure_torch()

        try:
            import torch.nn.functional as F
        except Exception as exc:
            raise AvocadoException(
                "PyTorch is required for TransformerClassifier but is not installed."
            ) from exc

        if self.model is None:
            raise AvocadoException("Model has not been trained yet.")

        cont_features, band_ids, mask, global_feats = self._parse_features(dataset)

        Xc = torch.tensor(cont_features, dtype=torch.float32).to(self.device)
        Xb = torch.tensor(band_ids, dtype=torch.long).to(self.device)
        M  = torch.tensor(mask, dtype=torch.float32).to(self.device)
        G  = torch.tensor(global_feats, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(Xc, Xb, M, G)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        predictions = pd.DataFrame(
            probs,
            index=dataset.metadata.index,
            columns=self.class_names,
        )
        predictions.index.name = "object_id"

        return predictions