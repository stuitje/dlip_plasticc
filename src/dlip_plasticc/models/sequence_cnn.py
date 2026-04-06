from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from avocado.classifier import Classifier
from avocado.settings import settings
from avocado.utils import AvocadoException

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
                "PyTorch is required for SequenceCNNClassifier but is not installed."
            ) from _TORCH_IMPORT_ERROR

    class _NNFallback:
        Module = _FallbackModule

    nn = _NNFallback()


def _choose_num_groups(num_channels, max_groups=8):
    """Choose a GroupNorm group count that divides num_channels."""
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class _ResidualTemporalBlock(nn.Module):
    """Residual dilated 1D conv block with GroupNorm."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        dropout=0.35,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length padding.")

        padding = dilation * (kernel_size - 1) // 2
        groups = _choose_num_groups(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, mask=None):
        # x: (B, C, T)
        # mask: (B, 1, T) with 1 for valid steps, 0 for padding
        residual = self.skip(x)

        out = self.conv1(x)
        if mask is not None:
            out = out * mask
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        if mask is not None:
            out = out * mask
        out = self.norm2(out)

        out = out + residual
        out = self.act2(out)
        out = self.drop2(out)

        if mask is not None:
            out = out * mask

        return out


class _MaskedMeanPool(nn.Module):
    """Masked mean pooling over time."""

    def forward(self, x, mask=None):
        # x: (B, C, T)
        if mask is None:
            return x.mean(dim=2)

        mask = mask.to(dtype=x.dtype)
        denom = mask.sum(dim=2).clamp_min(1.0)
        return (x * mask).sum(dim=2) / denom


class _SequenceCNNNet(nn.Module):
    """Regularized dilated residual CNN over padded light-curve sequences."""

    def __init__(
        self,
        cont_dim,
        num_classes,
        num_bands=6,
        global_dim=5,
        hidden_dim=64,
        band_emb_dim=8,
        kernel_size=3,
        dilations=(1, 2, 4),
        dropout=0.35,
        global_hidden_dim=32,
        head_hidden_dim=64,
    ):
        super().__init__()

        if band_emb_dim >= hidden_dim:
            raise ValueError("band_emb_dim must be smaller than hidden_dim.")

        cont_proj_dim = hidden_dim - band_emb_dim

        self.cont_norm = nn.LayerNorm(cont_dim)
        self.cont_proj = nn.Linear(cont_dim, cont_proj_dim)
        self.band_emb = nn.Embedding(num_bands, band_emb_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        blocks = []
        in_channels = hidden_dim
        for dilation in dilations:
            blocks.append(
                _ResidualTemporalBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = hidden_dim
        self.blocks = nn.ModuleList(blocks)

        self.pool = _MaskedMeanPool()

        self.global_mlp = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, global_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim + global_hidden_dim),
            nn.Linear(hidden_dim + global_hidden_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(self, cont_x, band_ids, mask=None, global_feats=None):
        # cont_x:   (B, T, C_cont)
        # band_ids: (B, T)
        # mask:     (B, T)
        # global:   (B, C_global)

        cont_x_norm = self.cont_norm(cont_x)
        x_cont = self.cont_proj(cont_x_norm)

        band_ids = torch.clamp(
            band_ids, min=0, max=self.band_emb.num_embeddings - 1
        )
        x_band = self.band_emb(band_ids)

        x = torch.cat([x_cont, x_band], dim=-1)   # (B, T, H)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        x = x.transpose(1, 2)                     # (B, H, T)

        mask_1d = None
        if mask is not None:
            mask_1d = mask.unsqueeze(1).to(dtype=x.dtype)
            x = x * mask_1d

        for block in self.blocks:
            x = block(x, mask_1d)

        pooled = self.pool(x, mask_1d)

        if global_feats is None:
            raise ValueError("global_feats must be provided.")

        global_repr = self.global_mlp(global_feats)
        fused = torch.cat([pooled, global_repr], dim=1)

        return self.head(fused)


class SequenceCNNClassifier(Classifier):
    """Regularized CNN classifier for light-curve sequences.

    Supported featurizer outputs
    ----------------------------
    1. Legacy form:
       - features with shape (n_samples, seq_len, channels)
       - or (features, mask)

       Expected legacy channel convention:
         [time, flux, flux_error, detected, band_id]

       This classifier will automatically derive:
         - log_dt from time
         - snr from flux / flux_error
         - band embedding from band_id

    2. Richer form:
       - (cont_features, band_ids, mask)
       - (cont_features, band_ids, mask, global_features)

    Notes
    -----
    - Uses band embeddings rather than scalar band IDs.
    - Uses residual dilated temporal convolutions.
    - Uses masked mean pooling only.
    - Derives lightweight global summary features if they are not provided.
    - Uses weighted loss for training but unweighted loss for validation.
    - Uses label smoothing during training when supported by the installed
      PyTorch version.
    """

    def __init__(
        self,
        name,
        featurizer,
        num_epochs=20,
        batch_size=64,
        lr=1e-3,
        class_weights=None,
        auto_class_weights=False,
        class_weight_power=0.5,
        label_smoothing=0.05,
        device=None,
        hidden_dim=64,
        band_emb_dim=8,
        kernel_size=3,
        dilations=(1, 2, 4),
        dropout=0.35,
        num_bands=6,
        global_hidden_dim=32,
        head_hidden_dim=64,
    ):
        super().__init__(name)

        self.featurizer = featurizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.class_weights = class_weights
        self.auto_class_weights = auto_class_weights
        self.class_weight_power = class_weight_power
        self.label_smoothing = label_smoothing
        self.device = device

        self.hidden_dim = hidden_dim
        self.band_emb_dim = band_emb_dim
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.dropout = dropout
        self.num_bands = num_bands
        self.global_hidden_dim = global_hidden_dim
        self.head_hidden_dim = head_hidden_dim

        self.model = None
        self.class_names = None
        self.history = None
        self.best_val_loss = None
        self.val_fold = None

    def _ensure_torch(self):
        if _TORCH_IMPORT_ERROR is not None:
            raise AvocadoException(
                "PyTorch is required for SequenceCNNClassifier but is not installed."
            ) from _TORCH_IMPORT_ERROR

    def _build_class_weight_tensor(self, class_names, class_indices_train, device):
        """Return a 1-D class weight tensor aligned to class_names order."""
        if not self.auto_class_weights and self.class_weights is None:
            return None

        weights = np.ones(len(class_names), dtype=np.float32)

        if self.auto_class_weights:
            counts = np.bincount(
                class_indices_train, minlength=len(class_names)
            ).astype(np.float32)
            counts = np.where(counts == 0, 1.0, counts)
            auto_weights = counts ** (-float(self.class_weight_power))
            auto_weights /= auto_weights.mean()
            weights *= auto_weights

        if self.class_weights is not None:
            for i, c in enumerate(class_names):
                weights[i] *= self.class_weights.get(c, 1.0)

        return torch.tensor(weights, dtype=torch.float32).to(device)

    def _derive_global_features(self, cont_features, mask):
        """Derive lightweight per-object global summary features.

        Output shape: (n_samples, 5)

        cont_features convention:
        [time, log_dt, flux, flux_err, snr, detected, ...]
        """
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
            [
                valid_frac,
                mean_abs_flux,
                mean_flux_err,
                max_abs_snr,
                mean_detected,
            ],
            axis=1,
        ).astype(np.float32)

        return global_feats

    def _compute_classification_metrics(self, y_true, y_pred, num_classes):
        """Compute accuracy, macro-F1, and balanced accuracy without sklearn."""
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)

        acc = float((y_true == y_pred).mean()) if len(y_true) > 0 else np.nan

        f1s = []
        recalls = []

        for c in range(num_classes):
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                f1 = 2.0 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            f1s.append(f1)
            recalls.append(recall)

        macro_f1 = float(np.mean(f1s)) if len(f1s) > 0 else np.nan
        balanced_acc = float(np.mean(recalls)) if len(recalls) > 0 else np.nan

        return {
            "acc": acc,
            "macro_f1": macro_f1,
            "balanced_acc": balanced_acc,
        }

    def _parse_features(self, dataset):
        features = dataset.select_features(self.featurizer)

        cont_features = None
        band_ids = None
        mask = None
        global_feats = None

        # Richer forms
        if isinstance(features, (tuple, list)):
            if len(features) == 4:
                cont_features, band_ids, mask, global_feats = features
            elif len(features) == 3:
                cont_features, band_ids, mask = features
            elif len(features) == 2:
                # Legacy form: (features, mask)
                features, mask = features
            else:
                raise AvocadoException(
                    "Unsupported feature tuple format for SequenceCNNClassifier."
                )

        # Legacy direct 3D array form
        if cont_features is None and band_ids is None:
            if hasattr(features, "values"):
                features = features.values
            features = np.asarray(features, dtype=float)

            if np.isnan(features).any() or np.isinf(features).any():
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            if features.ndim != 3:
                raise AvocadoException(
                    "SequenceCNNClassifier expects features shaped "
                    "(n, seq_len, channels), (features, mask), "
                    "(cont_features, band_ids, mask), or "
                    "(cont_features, band_ids, mask, global_features)."
                )

            if features.shape[2] < 5:
                raise AvocadoException(
                    "Legacy sequence features must have at least 5 channels: "
                    "[time, flux, flux_error, detected, band_id]."
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
                raise AvocadoException(
                    "cont_features must have shape (n, seq_len, channels)."
                )
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
        weight_decay=5e-2,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=2,
        min_lr=1e-6,
        early_stopping_patience=3,
    ):
        self._ensure_torch()

        try:
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as exc:
            raise AvocadoException(
                "PyTorch is required for SequenceCNNClassifier but is not installed."
            ) from exc

        if num_folds is None:
            num_folds = settings["num_folds"]
        if random_state is None:
            random_state = settings["fold_random_state"]

        if num_folds < 2:
            raise AvocadoException("SequenceCNNClassifier requires num_folds >= 2.")

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
        M_train = torch.tensor(mask[train_mask], dtype=torch.float32)
        G_train = torch.tensor(global_feats[train_mask], dtype=torch.float32)
        y_train = torch.tensor(class_indices[train_mask], dtype=torch.long)

        Xc_val = torch.tensor(cont_features[val_mask], dtype=torch.float32)
        Xb_val = torch.tensor(band_ids[val_mask], dtype=torch.long)
        M_val = torch.tensor(mask[val_mask], dtype=torch.float32)
        G_val = torch.tensor(global_feats[val_mask], dtype=torch.float32)
        y_val = torch.tensor(class_indices[val_mask], dtype=torch.long)

        train_ds = TensorDataset(Xc_train, Xb_train, M_train, G_train, y_train)
        val_ds = TensorDataset(Xc_val, Xb_val, M_val, G_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        num_bands_used = max(self.num_bands, int(np.max(band_ids)) + 1)

        model = _SequenceCNNNet(
            cont_dim=cont_features.shape[2],
            num_classes=len(class_names),
            num_bands=num_bands_used,
            global_dim=global_feats.shape[1],
            hidden_dim=self.hidden_dim,
            band_emb_dim=self.band_emb_dim,
            kernel_size=self.kernel_size,
            dilations=self.dilations,
            dropout=self.dropout,
            global_hidden_dim=self.global_hidden_dim,
            head_hidden_dim=self.head_hidden_dim,
        ).to(device)

        weight_tensor = self._build_class_weight_tensor(
            class_names, class_indices[train_mask], device
        )

        try:
            train_loss_fn = nn.CrossEntropyLoss(
                weight=weight_tensor,
                label_smoothing=float(self.label_smoothing),
            )
        except TypeError:
            train_loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

        val_loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=weight_decay,
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
            model.train()
            train_loss = 0.0

            for xc, xb, mb, gb, yb in train_loader:
                xc = xc.to(device)
                xb = xb.to(device)
                mb = mb.to(device)
                gb = gb.to(device)
                yb = yb.to(device)

                logits = model(xc, xb, mb, gb)
                loss = train_loss_fn(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * xc.size(0)

            train_loss /= len(train_ds)

            model.eval()
            val_loss = 0.0
            all_val_preds = []
            all_val_targets = []

            with torch.no_grad():
                for xc, xb, mb, gb, yb in val_loader:
                    xc = xc.to(device)
                    xb = xb.to(device)
                    mb = mb.to(device)
                    gb = gb.to(device)
                    yb = yb.to(device)

                    logits = model(xc, xb, mb, gb)
                    loss = val_loss_fn(logits, yb)
                    val_loss += loss.item() * xc.size(0)

                    preds = torch.argmax(logits, dim=1)
                    all_val_preds.append(preds.cpu().numpy())
                    all_val_targets.append(yb.cpu().numpy())

            val_loss /= len(val_ds)

            y_true = np.concatenate(all_val_targets) if all_val_targets else np.array([])
            y_pred = np.concatenate(all_val_preds) if all_val_preds else np.array([])

            metrics = self._compute_classification_metrics(
                y_true=y_true,
                y_pred=y_pred,
                num_classes=len(class_names),
            )

            val_acc = metrics["acc"]
            val_macro_f1 = metrics["macro_f1"]
            val_balanced_acc = metrics["balanced_acc"]

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_macro_f1": val_macro_f1,
                    "val_balanced_acc": val_balanced_acc,
                    "lr": current_lr,
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
                "val_acc: %.5f val_macro_f1: %.5f val_bal_acc: %.5f lr: %.6e"
                % (
                    epoch + 1,
                    train_loss,
                    val_loss,
                    val_acc,
                    val_macro_f1,
                    val_balanced_acc,
                    current_lr,
                )
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
        self.history = pd.DataFrame(history)
        self.best_val_loss = best_val_loss
        self.val_fold = val_fold

        print("Best validation loss: %.5f" % best_val_loss)

        return model

    def predict(self, dataset, show_progress=True):
        self._ensure_torch()

        try:
            import torch.nn.functional as F
        except Exception as exc:
            raise AvocadoException(
                "PyTorch is required for SequenceCNNClassifier but is not installed."
            ) from exc

        if self.model is None:
            raise AvocadoException("Model has not been trained yet.")

        cont_features, band_ids, mask, global_feats = self._parse_features(dataset)

        Xc = torch.tensor(cont_features, dtype=torch.float32).to(self.device)
        Xb = torch.tensor(band_ids, dtype=torch.long).to(self.device)
        M = torch.tensor(mask, dtype=torch.float32).to(self.device)
        G = torch.tensor(global_feats, dtype=torch.float32).to(self.device)

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