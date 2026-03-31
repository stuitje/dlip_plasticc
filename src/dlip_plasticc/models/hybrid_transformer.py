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
                "PyTorch is required for HybridTransformerClassifier but is not installed."
            ) from _TORCH_IMPORT_ERROR

    class _NNFallback:
        Module = _FallbackModule

    nn = _NNFallback()


class _TimePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding driven by actual time values."""

    def __init__(self, d_model):
        super().__init__()
        half_dim = d_model // 2
        i = torch.arange(0, half_dim, dtype=torch.float32)
        self.register_buffer(
            "div_term",
            torch.exp(i * (-np.log(10000.0) / d_model)),
        )
        self.d_model = d_model

    def forward(self, times):
        # times: (B, T)
        t = times.unsqueeze(-1) * 10000.0
        pe = torch.cat(
            [torch.sin(t * self.div_term), torch.cos(t * self.div_term)],
            dim=-1,
        )
        if pe.shape[-1] < self.d_model:
            pad = torch.zeros(
                pe.shape[0], pe.shape[1], self.d_model - pe.shape[-1],
                device=pe.device, dtype=pe.dtype
            )
            pe = torch.cat([pe, pad], dim=-1)
        return pe


class _AttentionPool(nn.Module):
    """Masked attention pooling over sequence embeddings."""

    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        # x: (B, T, D)
        scores = self.score(x).squeeze(-1)  # (B, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled


class _SequenceTransformerBranch(nn.Module):
    """Transformer branch for light-curve sequences."""

    def __init__(
        self,
        cont_dim,
        d_model=128,
        band_emb_dim=16,
        num_bands=6,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
    ):
        super().__init__()

        if band_emb_dim >= d_model:
            raise ValueError("band_emb_dim must be smaller than d_model.")

        cont_proj_dim = d_model - band_emb_dim

        self.cont_norm = nn.LayerNorm(cont_dim)
        self.cont_proj = nn.Linear(cont_dim, cont_proj_dim)
        self.band_emb = nn.Embedding(num_bands, band_emb_dim)

        self.input_norm = nn.LayerNorm(d_model)
        self.time_pe = _TimePositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = _AttentionPool(d_model)

    def forward(self, cont_x, band_ids, mask=None):
        # cont_x:   (B, T, C_cont)
        # band_ids: (B, T)
        # mask:     (B, T)
        times = cont_x[:, :, 0]

        cont_x = self.cont_norm(cont_x)
        x_cont = self.cont_proj(cont_x)

        band_ids = torch.clamp(
            band_ids,
            min=0,
            max=self.band_emb.num_embeddings - 1,
        )
        x_band = self.band_emb(band_ids)

        x = torch.cat([x_cont, x_band], dim=-1)
        x = self.input_norm(x)
        x = self.dropout(x + self.time_pe(times))

        pad_mask = None
        if mask is not None:
            pad_mask = mask == 0

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.pool(x, mask)

        return x


class _FeatureMLPBranch(nn.Module):
    """MLP branch for Boone/Avocado tabular features."""

    def __init__(self, input_dim, output_dim=128, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class _HybridTransformerNet(nn.Module):
    """Fusion model: Transformer sequence branch + feature MLP branch."""

    def __init__(
        self,
        cont_dim,
        feature_dim,
        num_classes,
        d_model=128,
        band_emb_dim=16,
        num_bands=6,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
    ):
        super().__init__()

        self.sequence_branch = _SequenceTransformerBranch(
            cont_dim=cont_dim,
            d_model=d_model,
            band_emb_dim=band_emb_dim,
            num_bands=num_bands,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.feature_branch = _FeatureMLPBranch(
            input_dim=feature_dim,
            output_dim=d_model,
            dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, cont_x, band_ids, mask, features):
        seq_emb = self.sequence_branch(cont_x, band_ids, mask)
        feat_emb = self.feature_branch(features)
        fused = torch.cat([seq_emb, feat_emb], dim=1)
        return self.head(fused)


class HybridTransformerClassifier(Classifier):
    """Hybrid classifier combining raw light-curve sequences and Boone features.

    Parameters
    ----------
    sequence_featurizer
        Featurizer producing sequence-style features. Works with the current
        `PlasticcSequenceFeaturizer` legacy output `(features, mask)`, where
        the feature channels are expected to be:
            [time, flux, flux_error, detected, band_id]
    feature_featurizer
        Boone/Avocado-style feature featurizer for tabular features.

    Notes
    -----
    - The dataset must have `objects` loaded because the sequence branch needs
      access to light-curve observations.
    - For the tabular branch:
        * if `dataset.raw_features` is already present, it is assumed to
          correspond to `feature_featurizer`
        * otherwise, raw features are computed on the fly
    - Tabular preprocessing is fit on the training split only, which avoids
      the leakage bug from the old code.
    """

    def __init__(
        self,
        name,
        sequence_featurizer,
        feature_featurizer,
        num_epochs=20,
        batch_size=64,
        lr=1e-4,
        class_weights=None,
        device=None,
        d_model=128,
        band_emb_dim=16,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.2,
        num_bands=6,
    ):
        super().__init__(name)

        self.sequence_featurizer = sequence_featurizer
        self.feature_featurizer = feature_featurizer

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.class_weights = class_weights
        self.device = device

        self.d_model = d_model
        self.band_emb_dim = band_emb_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_bands = num_bands

        self.model = None
        self.class_names = None
        self.best_val_loss = None
        self.history = None
        self.val_fold = None

        # Saved tabular preprocessing state
        self._feature_base_columns = None
        self._feature_missing_cols = None
        self._feature_final_columns = None
        self._feature_medians = None
        self._feature_means = None
        self._feature_stds = None

    def _ensure_torch(self):
        if _TORCH_IMPORT_ERROR is not None:
            raise AvocadoException(
                "PyTorch is required for HybridTransformerClassifier but is not installed."
            ) from _TORCH_IMPORT_ERROR

    # ------------------------------------------------------------------
    # Sequence handling
    # ------------------------------------------------------------------
    def _extract_sequence_raw_features(self, dataset):
        if dataset.objects is None:
            raise AvocadoException(
                "HybridTransformerClassifier requires dataset.objects to be loaded "
                "for the sequence branch. Load the dataset with observations."
            )

        rows = []
        object_ids = []

        for obj in dataset.objects:
            raw = self.sequence_featurizer.extract_raw_features(obj)
            if isinstance(raw, tuple):
                raw = raw[0]
            rows.append(raw)
            object_ids.append(obj.metadata["object_id"])

        raw_df = pd.DataFrame(rows, index=object_ids)
        raw_df.index.name = "object_id"
        raw_df = raw_df.loc[dataset.metadata.index]

        return raw_df

    def _parse_sequence_features(self, dataset):
        raw_df = self._extract_sequence_raw_features(dataset)
        features = self.sequence_featurizer.select_features(raw_df)

        mask = None
        if isinstance(features, (tuple, list)) and len(features) == 2:
            features, mask = features

        if hasattr(features, "values"):
            features = features.values
        features = np.asarray(features, dtype=float)

        if np.isnan(features).any() or np.isinf(features).any():
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if features.ndim != 3:
            raise AvocadoException(
                "Sequence featurizer must produce features shaped "
                "(n, seq_len, channels) or (features, mask)."
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

        # Legacy convention:
        # [time, flux, flux_error, detected, band_id]
        time_chan = features[:, :, 0].astype(np.float32)
        flux_chan = features[:, :, 1].astype(np.float32)
        flux_err_chan = features[:, :, 2].astype(np.float32)
        detected_chan = features[:, :, 3].astype(np.float32)
        band_ids = np.rint(features[:, :, -1]).astype(np.int64)

        # Derive log delta-time from time
        dt = np.diff(time_chan, axis=1, prepend=time_chan[:, :1])
        dt = np.clip(dt, 0.0, None)
        log_dt = np.log1p(dt).astype(np.float32)

        # Derive clipped S/N
        snr = flux_chan / (np.abs(flux_err_chan) + 1e-8)
        snr = np.clip(snr, -20.0, 20.0).astype(np.float32)

        # Include any extra continuous channels between detected and band_id
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

        return cont_features, band_ids.astype(np.int64), mask.astype(np.float32)

    # ------------------------------------------------------------------
    # Tabular feature handling
    # ------------------------------------------------------------------
    def _get_tabular_feature_frame(self, dataset):
        if dataset.raw_features is None:
            if dataset.objects is None:
                raise AvocadoException(
                    "dataset.raw_features is missing and dataset.objects is not loaded. "
                    "Cannot compute Boone-style features on the fly."
                )
            dataset.extract_raw_features(self.feature_featurizer)

        selected = self.feature_featurizer.select_features(dataset.raw_features)

        if isinstance(selected, pd.DataFrame):
            feature_df = selected.copy()
        elif isinstance(selected, dict):
            feature_df = pd.DataFrame(selected, index=dataset.metadata.index)
        else:
            arr = np.asarray(selected, dtype=float)
            if arr.ndim != 2:
                raise AvocadoException(
                    "Feature featurizer must produce a 2D feature matrix or DataFrame."
                )
            feature_df = pd.DataFrame(
                arr,
                index=dataset.metadata.index,
                columns=[f"feature_{i}" for i in range(arr.shape[1])],
            )

        feature_df = feature_df.reindex(dataset.metadata.index)

        return feature_df

    def _fit_tabular_preprocessor(self, feature_df):
        feature_df = feature_df.copy()

        self._feature_base_columns = list(feature_df.columns)
        self._feature_missing_cols = [
            col for col in feature_df.columns if feature_df[col].isnull().any()
        ]

        indicators = pd.DataFrame(
            {
                f"{col}_missing": feature_df[col].isnull().astype(float)
                for col in self._feature_missing_cols
            },
            index=feature_df.index,
        )

        feature_aug = pd.concat([feature_df, indicators], axis=1)
        self._feature_final_columns = list(feature_aug.columns)

        self._feature_medians = feature_aug.median()
        feature_imp = feature_aug.fillna(self._feature_medians)

        self._feature_means = feature_imp.mean()
        self._feature_stds = feature_imp.std(ddof=0).replace(0, 1.0)

    def _transform_tabular_features(self, feature_df):
        if self._feature_base_columns is None:
            raise AvocadoException(
                "Tabular feature preprocessor is not fit. Train the model first."
            )

        feature_df = feature_df.copy()
        feature_df = feature_df.reindex(columns=self._feature_base_columns)

        indicators = pd.DataFrame(
            {
                f"{col}_missing": feature_df[col].isnull().astype(float)
                for col in self._feature_missing_cols
            },
            index=feature_df.index,
        )

        feature_aug = pd.concat([feature_df, indicators], axis=1)
        feature_aug = feature_aug.reindex(columns=self._feature_final_columns)

        feature_imp = feature_aug.fillna(self._feature_medians)
        feature_scaled = (feature_imp - self._feature_means) / self._feature_stds

        feature_scaled = feature_scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        return feature_scaled.values.astype(np.float32)

    # ------------------------------------------------------------------
    # Training / prediction
    # ------------------------------------------------------------------
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
                "PyTorch is required for HybridTransformerClassifier but is not installed."
            ) from exc

        if num_folds is None:
            num_folds = settings["num_folds"]
        if random_state is None:
            random_state = settings["fold_random_state"]

        if num_folds < 2:
            raise AvocadoException("HybridTransformerClassifier requires num_folds >= 2.")

        cont_features, band_ids, mask = self._parse_sequence_features(dataset)
        tabular_df = self._get_tabular_feature_frame(dataset)

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

        # Fit tabular preprocessing on train split only
        self._fit_tabular_preprocessor(tabular_df.loc[train_mask])
        tabular_features = self._transform_tabular_features(tabular_df)

        Xc_train = torch.tensor(cont_features[train_mask], dtype=torch.float32)
        Xb_train = torch.tensor(band_ids[train_mask], dtype=torch.long)
        M_train = torch.tensor(mask[train_mask], dtype=torch.float32)
        Xt_train = torch.tensor(tabular_features[train_mask], dtype=torch.float32)
        y_train = torch.tensor(class_indices[train_mask], dtype=torch.long)

        Xc_val = torch.tensor(cont_features[val_mask], dtype=torch.float32)
        Xb_val = torch.tensor(band_ids[val_mask], dtype=torch.long)
        M_val = torch.tensor(mask[val_mask], dtype=torch.float32)
        Xt_val = torch.tensor(tabular_features[val_mask], dtype=torch.float32)
        y_val = torch.tensor(class_indices[val_mask], dtype=torch.long)

        train_ds = TensorDataset(Xc_train, Xb_train, M_train, Xt_train, y_train)
        val_ds = TensorDataset(Xc_val, Xb_val, M_val, Xt_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        num_bands_used = max(self.num_bands, int(np.max(band_ids)) + 1)

        model = _HybridTransformerNet(
            cont_dim=cont_features.shape[2],
            feature_dim=tabular_features.shape[1],
            num_classes=len(class_names),
            d_model=self.d_model,
            band_emb_dim=self.band_emb_dim,
            num_bands=num_bands_used,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(device)

        if self.class_weights is not None:
            weight_list = [self.class_weights.get(c, 1.0) for c in class_names]
            weight = torch.tensor(weight_list, dtype=torch.float32).to(device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()

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
            # ---- train ----
            model.train()
            train_loss = 0.0

            for xc, xb, mb, xt, yb in train_loader:
                xc = xc.to(device)
                xb = xb.to(device)
                mb = mb.to(device)
                xt = xt.to(device)
                yb = yb.to(device)

                logits = model(xc, xb, mb, xt)
                loss = loss_fn(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * xc.size(0)

            train_loss /= len(train_ds)

            # ---- validation ----
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for xc, xb, mb, xt, yb in val_loader:
                    xc = xc.to(device)
                    xb = xb.to(device)
                    mb = mb.to(device)
                    xt = xt.to(device)
                    yb = yb.to(device)

                    logits = model(xc, xb, mb, xt)
                    loss = loss_fn(logits, yb)

                    val_loss += loss.item() * xc.size(0)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)

            val_loss /= len(val_ds)
            val_acc = correct / total if total > 0 else np.nan

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
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
                "PyTorch is required for HybridTransformerClassifier but is not installed."
            ) from exc

        if self.model is None:
            raise AvocadoException("Model has not been trained yet.")

        cont_features, band_ids, mask = self._parse_sequence_features(dataset)
        tabular_df = self._get_tabular_feature_frame(dataset)
        tabular_features = self._transform_tabular_features(tabular_df)

        Xc = torch.tensor(cont_features, dtype=torch.float32).to(self.device)
        Xb = torch.tensor(band_ids, dtype=torch.long).to(self.device)
        M = torch.tensor(mask, dtype=torch.float32).to(self.device)
        Xt = torch.tensor(tabular_features, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(Xc, Xb, M, Xt)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        predictions = pd.DataFrame(
            probs,
            index=dataset.metadata.index,
            columns=self.class_names,
        )
        predictions.index.name = "object_id"

        return predictions