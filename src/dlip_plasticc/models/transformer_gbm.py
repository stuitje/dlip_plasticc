from __future__ import annotations

"""TransformerGBMClassifier
===========================
A two-stage hybrid classifier for PLAsTiCC / avocado:

  Stage 1 — Train a lightweight sequence transformer on raw light-curve
             sequences (cross-entropy, fold-based, early stopping).
             The final classification head is discarded after training.

  Stage 2 — Extract per-object transformer embeddings (attention-pooled
             sequence representation, shape d_model) and concatenate them
             with the Boone/avocado GP tabular features (41 features).
             Train LightGBM on this combined feature matrix, optimising
             the PLAsTiCC flat-weighted multiclass log-loss.

This approach is more robust to train/test distribution shift than an
end-to-end joint model because:
  - LightGBM is inherently more resistant to covariate shift than a deep
    classification head.
  - The GP tabular features are already smoothed / denoised by the GP fit,
    making them more stable across the training / test redshift gap.
  - The transformer embeddings add temporal-structure information that the
    tabular features cannot fully capture.

Usage
-----
    from dlip_plasticc.models import TransformerGBMClassifier
    from dlip_plasticc.features import PlasticcSequenceFeaturizer
    import avocado

    seq_featurizer  = PlasticcSequenceFeaturizer(seq_len=350)
    gp_featurizer   = avocado.plasticc.PlasticcFeaturizer()

    clf = TransformerGBMClassifier(
        name="transformer_gbm_v1",
        sequence_featurizer=seq_featurizer,
        gp_featurizer=gp_featurizer,
    )

    dataset = avocado.load("plasticc_train", num_chunks=1)

    # Optional: provide precomputed raw features explicitly
    seq_raw = ...
    gp_raw  = ...

    clf.train(dataset, seq_raw=seq_raw, gp_raw=gp_raw)
    clf.write(overwrite=True)

    test_dataset = avocado.load("plasticc_test", chunk=0, num_chunks=21)
    preds = clf.predict(test_dataset, seq_raw=test_seq_raw, gp_raw=test_gp_raw)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from avocado.classifier import Classifier
from avocado.settings import settings
from avocado.utils import AvocadoException

# --------------------------------------------------------------------------- #
# Optional imports                                                             #
# --------------------------------------------------------------------------- #
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
                "PyTorch is required for TransformerGBMClassifier."
            ) from _TORCH_IMPORT_ERROR

    class _NNFallback:
        Module = _FallbackModule

    nn = _NNFallback()

_LGBM_IMPORT_ERROR = None
try:
    import lightgbm as lgb
except Exception as exc:
    lgb = None
    _LGBM_IMPORT_ERROR = exc


# --------------------------------------------------------------------------- #
# PyTorch modules (sequence encoder only, no classification head)             #
# --------------------------------------------------------------------------- #

class _TimePositionalEncoding(nn.Module):
    """Sinusoidal encoding driven by actual (normalised) time values."""

    def __init__(self, d_model: int):
        super().__init__()
        half = d_model // 2
        i = torch.arange(0, half, dtype=torch.float32)
        self.register_buffer(
            "div_term",
            torch.exp(i * (-np.log(10000.0) / d_model)),
        )
        self.d_model = d_model

    def forward(self, times: "torch.Tensor") -> "torch.Tensor":
        # times: (B, T)
        t = times.unsqueeze(-1) * 10000.0
        pe = torch.cat(
            [torch.sin(t * self.div_term), torch.cos(t * self.div_term)],
            dim=-1,
        )
        if pe.shape[-1] < self.d_model:
            pad = torch.zeros(
                pe.shape[0], pe.shape[1], self.d_model - pe.shape[-1],
                device=pe.device, dtype=pe.dtype,
            )
            pe = torch.cat([pe, pad], dim=-1)
        return pe


class _AttentionPool(nn.Module):
    """Masked soft-attention pooling → single vector per sequence."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(
        self,
        x: "torch.Tensor",
        mask: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        scores = self.score(x).squeeze(-1)            # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)        # (B, T)
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)


class _SequenceEncoder(nn.Module):
    """Transformer encoder that returns a pooled embedding, NOT class logits.

    The classification head is intentionally absent — we feed the pooled
    embedding to LightGBM instead.
    """

    def __init__(
        self,
        cont_dim: int,
        num_classes: int,          # needed only during stage-1 training
        d_model: int = 128,
        band_emb_dim: int = 16,
        num_bands: int = 6,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
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

        # Lightweight classification head used only during stage-1 training.
        # We discard it before stage 2 but keep it here so the module can be
        # trained end-to-end with a standard cross-entropy loss.
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def encode(
        self,
        cont_x: "torch.Tensor",
        band_ids: "torch.Tensor",
        mask: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Return the pooled embedding (B, d_model) without the head."""
        times = cont_x[:, :, 0]

        cont_x = self.cont_norm(cont_x)
        x_cont = self.cont_proj(cont_x)

        band_ids = torch.clamp(band_ids, 0, self.band_emb.num_embeddings - 1)
        x_band = self.band_emb(band_ids)

        x = torch.cat([x_cont, x_band], dim=-1)
        x = self.input_norm(x)
        x = self.dropout(x + self.time_pe(times))

        pad_mask = (mask == 0) if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.pool(x, mask)

    def forward(
        self,
        cont_x: "torch.Tensor",
        band_ids: "torch.Tensor",
        mask: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Forward pass used during stage-1 training → class logits."""
        emb = self.encode(cont_x, band_ids, mask)
        return self.head(emb)


# --------------------------------------------------------------------------- #
# Classifier                                                                  #
# --------------------------------------------------------------------------- #

class TransformerGBMClassifier(Classifier):
    """Two-stage hybrid: sequence transformer embeddings + LightGBM.

    Parameters
    ----------
    name
        Avocado classifier name (used for serialisation).
    sequence_featurizer
        Featurizer producing ``(features, mask)`` where features has shape
        ``(n_samples, seq_len, channels)`` with channels ordered as:
        ``[time, flux, flux_error, detected, band_id]``.
        Compatible with ``PlasticcSequenceFeaturizer``.
    gp_featurizer
        Boone / avocado GP-based featurizer for the 41 tabular features.
    transformer_epochs
        Maximum epochs for stage-1 transformer training.
    transformer_batch_size
        Batch size for stage-1 training.
    transformer_lr
        Learning rate for AdamW in stage 1.
    d_model
        Transformer hidden dimension (also the embedding dimension fed to LGBM).
    band_emb_dim
        Dimension of the band embedding inside the transformer.
    nhead
        Number of attention heads.
    num_layers
        Number of transformer encoder layers.
    dim_feedforward
        Feed-forward dimension inside each encoder layer.
    dropout
        Dropout rate applied in both the transformer and stage-2 LGBM (via
        feature fraction / bagging).
    num_bands
        Number of photometric bands (6 for PLAsTiCC LSST).
    lgbm_params
        Dict of LightGBM parameters. Sensible defaults tuned for PLAsTiCC
        flat-weighted logloss are provided; pass overrides here.
    class_weights
        Optional dict mapping class label → weight for the transformer
        cross-entropy loss in stage 1. Not used for LightGBM (the
        flat-weighted metric normalises by class automatically).
    device
        PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``. Auto-detected
        if ``None``.
    """

    # Default LightGBM hyperparameters. These mirror Boone (2019) Table 4
    # where applicable, with adjustments for the richer feature set.
    _DEFAULT_LGBM_PARAMS: dict = {
        "objective": "multiclass",
        "boosting_type": "gbdt",
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "subsample_freq": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "min_split_gain": 10.0,
        "min_child_weight": 2000,
        "max_depth": 7,
        "num_leaves": 50,
        "n_estimators": 2000,
        "early_stopping_rounds": 50,
        "verbose": -1,
    }

    def __init__(
        self,
        name: str,
        sequence_featurizer,
        gp_featurizer,
        transformer_epochs: int = 15,
        transformer_batch_size: int = 64,
        transformer_lr: float = 1e-4,
        d_model: int = 128,
        band_emb_dim: int = 16,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        num_bands: int = 6,
        lgbm_params: dict | None = None,
        class_weights: dict | None = None,
        auto_class_weights: bool = False,
        device: str | None = None,
    ):
        super().__init__(name)

        self.sequence_featurizer = sequence_featurizer
        self.gp_featurizer = gp_featurizer

        self.transformer_epochs = transformer_epochs
        self.transformer_batch_size = transformer_batch_size
        self.transformer_lr = transformer_lr

        self.d_model = d_model
        self.band_emb_dim = band_emb_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_bands = num_bands

        self.lgbm_params = {**self._DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}
        self.class_weights = class_weights
        self.auto_class_weights = auto_class_weights
        self.device = device

        # Set after training
        self.encoder: "_SequenceEncoder | None" = None
        self.gbm: "lgb.Booster | None" = None
        self.class_names: "np.ndarray | None" = None
        self.best_val_loss_transformer: float = np.inf
        self.history: "pd.DataFrame | None" = None

        # Tabular preprocessing state (fit on training split only)
        self._gp_base_columns: "list | None" = None
        self._gp_missing_cols: "list | None" = None
        self._gp_final_columns: "list | None" = None
        self._gp_medians: "pd.Series | None" = None
        self._gp_means: "pd.Series | None" = None
        self._gp_stds: "pd.Series | None" = None

    # ---------------------------------------------------------------------- #
    # Guards                                                                 #
    # ---------------------------------------------------------------------- #

    def _ensure_torch(self):
        if _TORCH_IMPORT_ERROR is not None:
            raise AvocadoException(
                "PyTorch is required for TransformerGBMClassifier."
            ) from _TORCH_IMPORT_ERROR

    def _ensure_lgbm(self):
        if _LGBM_IMPORT_ERROR is not None:
            raise AvocadoException(
                "LightGBM is required for TransformerGBMClassifier. "
                "Install it with: pip install lightgbm"
            ) from _LGBM_IMPORT_ERROR

    # ------------------------------------------------------------------ #
    # Class-weight helpers (shared by Stage 1 and Stage 2)               #
    # ------------------------------------------------------------------ #

    def _class_weight_array(self, class_names, class_indices_train):
        # Return a 1-D float32 array of per-class weights, or None.
        # auto_class_weights=True  -> inverse-frequency from training split,
        #   normalised so mean weight = 1.
        # class_weights dict       -> per-class multipliers (default 1.0).
        # Both can be combined multiplicatively.
        # Neither set              -> returns None (uniform weighting).
        if not self.auto_class_weights and self.class_weights is None:
            return None

        weights = np.ones(len(class_names), dtype=np.float32)

        if self.auto_class_weights:
            counts = np.bincount(
                class_indices_train, minlength=len(class_names)
            ).astype(np.float32)
            counts = np.where(counts == 0, 1.0, counts)
            inv_freq = 1.0 / counts
            inv_freq /= inv_freq.mean()   # normalise: mean weight = 1
            weights *= inv_freq

        if self.class_weights is not None:
            for i, c in enumerate(class_names):
                weights[i] *= self.class_weights.get(c, 1.0)

        return weights

    def _torch_weight_tensor(self, class_names, class_indices_train, device):
        # Torch tensor version for CrossEntropyLoss.
        w = self._class_weight_array(class_names, class_indices_train)
        if w is None:
            return None
        return torch.tensor(w, dtype=torch.float32).to(device)

    def _lgbm_sample_weights(self, class_names, class_indices_train):
        # Per-sample weights for lgb.Dataset (class weight -> each sample).
        w = self._class_weight_array(class_names, class_indices_train)
        if w is None:
            return None
        return w[class_indices_train]

    # ---------------------------------------------------------------------- #
    # Sequence feature parsing (reused from TransformerClassifier)           #
    # ---------------------------------------------------------------------- #

    def _get_seq_raw_features(self, dataset):
        """Extract sequence raw features, returning a DataFrame.

        Uses dataset.raw_features if it contains sequence data (has a
        'sequence' column), otherwise runs extraction via dataset.objects.
        Does NOT overwrite dataset.raw_features so the GP featurizer slot
        is left untouched.
        """
        raw = dataset.raw_features
        if raw is not None and hasattr(raw, "columns") and "sequence" in raw.columns:
            return raw

        if dataset.objects is None:
            raise AvocadoException(
                "Sequence raw features are not available and dataset.objects "
                "is not loaded. Call dataset.extract_raw_features(seq_featurizer) "
                "before training, or load the dataset with metadata_only=False."
            )

        rows, ids = [], []
        for obj in dataset.objects:
            raw_obj = self.sequence_featurizer.extract_raw_features(obj)
            if isinstance(raw_obj, tuple):
                raw_obj = raw_obj[0]
            rows.append(raw_obj)
            ids.append(obj.metadata["object_id"])

        seq_raw = pd.DataFrame(rows, index=ids)
        seq_raw.index.name = "object_id"
        return seq_raw.loc[dataset.metadata.index]

    def _parse_sequence_features(self, dataset, seq_raw=None):
        """Return (cont_features, band_ids, mask) as numpy arrays.

        Parameters
        ----------
        dataset
            Avocado dataset.
        seq_raw
            Pre-extracted sequence raw features DataFrame. If None, features
            are obtained via _get_seq_raw_features(dataset).
        """
        if seq_raw is None:
            seq_raw = self._get_seq_raw_features(dataset)

        features = self.sequence_featurizer.select_features(seq_raw)

        mask = None
        if isinstance(features, (tuple, list)) and len(features) == 2:
            features, mask = features

        if hasattr(features, "values"):
            features = features.values
        features = np.asarray(features, dtype=float)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if features.ndim != 3 or features.shape[2] < 5:
            raise AvocadoException(
                "Sequence featurizer must produce (n, seq_len, ≥5 channels)."
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
        log_dt = np.log1p(np.clip(dt, 0.0, None)).astype(np.float32)

        snr = np.clip(
            flux_chan / (np.abs(flux_err_chan) + 1e-8), -20.0, 20.0
        ).astype(np.float32)

        base_cont = [
            time_chan[:, :, None],
            log_dt[:, :, None],
            flux_chan[:, :, None],
            flux_err_chan[:, :, None],
            snr[:, :, None],
            detected_chan[:, :, None],
        ]
        if features.shape[2] > 5:
            base_cont.append(features[:, :, 4:-1].astype(np.float32))

        cont_features = np.concatenate(base_cont, axis=2).astype(np.float32)
        return cont_features, band_ids, mask.astype(np.float32)

    # ---------------------------------------------------------------------- #
    # GP tabular feature parsing                                             #
    # ---------------------------------------------------------------------- #

    def _get_gp_feature_frame(self, dataset, gp_raw=None) -> pd.DataFrame:
        """Return raw GP features as a DataFrame indexed by object_id.

        Parameters
        ----------
        dataset
            Avocado dataset.
        gp_raw
            Optional precomputed GP raw-feature table. When provided, this is
            used directly and dataset.raw_features is ignored.
        """
        raw = gp_raw if gp_raw is not None else dataset.raw_features

        if raw is None:
            if dataset.objects is None:
                raise AvocadoException(
                    "GP raw features are missing and dataset.objects is not "
                    "loaded. Cannot compute GP features on the fly."
                )
            dataset.extract_raw_features(self.gp_featurizer)
            raw = dataset.raw_features

        selected = self.gp_featurizer.select_features(raw)

        if isinstance(selected, pd.DataFrame):
            df = selected.copy()
        else:
            arr = np.asarray(selected, dtype=float)
            df = pd.DataFrame(
                arr,
                index=dataset.metadata.index,
                columns=[f"gp_{i}" for i in range(arr.shape[1])],
            )

        return df.reindex(dataset.metadata.index)

    def _fit_gp_preprocessor(self, gp_df: pd.DataFrame):
        """Fit imputation and z-score scaling on the training split."""
        gp_df = gp_df.copy()
        self._gp_base_columns = list(gp_df.columns)
        self._gp_missing_cols = [
            c for c in gp_df.columns if gp_df[c].isnull().any()
        ]

        indicators = pd.DataFrame(
            {f"{c}_missing": gp_df[c].isnull().astype(float)
             for c in self._gp_missing_cols},
            index=gp_df.index,
        )
        aug = pd.concat([gp_df, indicators], axis=1)
        self._gp_final_columns = list(aug.columns)
        self._gp_medians = aug.median()

        imp = aug.fillna(self._gp_medians)
        self._gp_means = imp.mean()
        self._gp_stds = imp.std(ddof=0).replace(0, 1.0)

    def _transform_gp_features(self, gp_df: pd.DataFrame) -> np.ndarray:
        """Apply saved imputation + z-score to a GP feature DataFrame."""
        if self._gp_base_columns is None:
            raise AvocadoException("GP preprocessor not fitted. Train first.")

        gp_df = gp_df.copy().reindex(columns=self._gp_base_columns)

        indicators = pd.DataFrame(
            {f"{c}_missing": gp_df[c].isnull().astype(float)
             for c in self._gp_missing_cols},
            index=gp_df.index,
        )
        aug = pd.concat([gp_df, indicators], axis=1)
        aug = aug.reindex(columns=self._gp_final_columns)

        imp = aug.fillna(self._gp_medians)
        scaled = (imp - self._gp_means) / self._gp_stds
        scaled = scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        return scaled.values.astype(np.float32)

    # ---------------------------------------------------------------------- #
    # Stage 1 — train sequence transformer                                   #
    # ---------------------------------------------------------------------- #

    def _train_transformer(
        self,
        cont_features: np.ndarray,
        band_ids: np.ndarray,
        mask: np.ndarray,
        class_indices: np.ndarray,
        class_names: np.ndarray,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        show_progress: bool,
        weight_decay: float,
        lr_scheduler_factor: float,
        lr_scheduler_patience: int,
        min_lr: float,
        early_stopping_patience: int,
    ) -> "_SequenceEncoder":
        from torch.utils.data import DataLoader, TensorDataset

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        Xc_tr = torch.tensor(cont_features[train_mask], dtype=torch.float32)
        Xb_tr = torch.tensor(band_ids[train_mask], dtype=torch.long)
        M_tr = torch.tensor(mask[train_mask], dtype=torch.float32)
        y_tr = torch.tensor(class_indices[train_mask], dtype=torch.long)

        Xc_va = torch.tensor(cont_features[val_mask], dtype=torch.float32)
        Xb_va = torch.tensor(band_ids[val_mask], dtype=torch.long)
        M_va = torch.tensor(mask[val_mask], dtype=torch.float32)
        y_va = torch.tensor(class_indices[val_mask], dtype=torch.long)

        train_ds = TensorDataset(Xc_tr, Xb_tr, M_tr, y_tr)
        val_ds = TensorDataset(Xc_va, Xb_va, M_va, y_va)

        train_loader = DataLoader(
            train_ds, batch_size=self.transformer_batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.transformer_batch_size, shuffle=False
        )

        num_bands_used = max(self.num_bands, int(np.max(band_ids)) + 1)

        encoder = _SequenceEncoder(
            cont_dim=cont_features.shape[2],
            num_classes=len(class_names),
            d_model=self.d_model,
            band_emb_dim=self.band_emb_dim,
            num_bands=num_bands_used,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(device)

        weight_tensor = self._torch_weight_tensor(
            class_names, class_indices[train_mask], device
        )
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

        optimizer = torch.optim.AdamW(
            encoder.parameters(), lr=self.transformer_lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_scheduler_factor,
            patience=lr_scheduler_patience, threshold=1e-4, min_lr=min_lr,
        )

        best_val_loss = np.inf
        best_state = None
        history = []
        no_improve = 0

        iterator = range(self.transformer_epochs)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="[Stage 1] Transformer epochs",
                dynamic_ncols=True,
            )

        for epoch in iterator:
            encoder.train()
            train_loss = 0.0
            for xc, xb, mb, yb in train_loader:
                xc, xb, mb, yb = (
                    xc.to(device), xb.to(device), mb.to(device), yb.to(device)
                )
                logits = encoder(xc, xb, mb)
                loss = loss_fn(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * xc.size(0)
            train_loss /= len(train_ds)

            encoder.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for xc, xb, mb, yb in val_loader:
                    xc, xb, mb, yb = (
                        xc.to(device), xb.to(device), mb.to(device), yb.to(device)
                    )
                    logits = encoder(xc, xb, mb)
                    val_loss += loss_fn(logits, yb).item() * xc.size(0)
                    correct += (logits.argmax(1) == yb).sum().item()
            val_loss /= len(val_ds)
            val_acc = correct / len(val_ds)

            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]

            history.append(
                dict(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    lr=lr,
                )
            )

            msg = (
                "Epoch %d  train=%.5f  val=%.5f  acc=%.4f  lr=%.2e"
                % (epoch + 1, train_loss, val_loss, val_acc, lr)
            )
            tqdm.write(msg) if show_progress else print(msg)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in encoder.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

            if early_stopping_patience and no_improve >= early_stopping_patience:
                msg = (
                    "Early stopping at epoch %d. Best val=%.5f"
                    % (epoch + 1, best_val_loss)
                )
                tqdm.write(msg) if show_progress else print(msg)
                break

        if best_state is not None:
            encoder.load_state_dict(best_state)

        self.best_val_loss_transformer = best_val_loss
        self.history = pd.DataFrame(history)
        print("[Stage 1] Best transformer val loss: %.5f" % best_val_loss)
        return encoder

    # ---------------------------------------------------------------------- #
    # Stage 2 — extract embeddings and train LightGBM                        #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def _extract_embeddings(
        self,
        encoder: "_SequenceEncoder",
        cont_features: np.ndarray,
        band_ids: np.ndarray,
        mask: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Run the encoder in eval mode and return pooled embeddings."""
        device = next(encoder.parameters()).device
        encoder.eval()

        n = cont_features.shape[0]
        out = np.zeros((n, self.d_model), dtype=np.float32)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xc = torch.tensor(cont_features[start:end], dtype=torch.float32).to(device)
            xb = torch.tensor(band_ids[start:end], dtype=torch.long).to(device)
            mb = torch.tensor(mask[start:end], dtype=torch.float32).to(device)
            out[start:end] = encoder.encode(xc, xb, mb).cpu().numpy()

        return out

    def _build_lgbm_features(
        self,
        embeddings: np.ndarray,
        gp_scaled: np.ndarray,
    ) -> np.ndarray:
        """Concatenate transformer embeddings and scaled GP features."""
        return np.concatenate([embeddings, gp_scaled], axis=1).astype(np.float32)

    def _train_lgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_names: np.ndarray,
        show_progress: bool,
    ) -> "lgb.Booster":
        params = {**self.lgbm_params, "num_class": len(class_names)}
        n_estimators = params.pop("n_estimators", 2000)
        early_stopping = params.pop("early_stopping_rounds", 50)

        sample_weights = self._lgbm_sample_weights(class_names, y_train)
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [lgb.log_evaluation(period=50 if show_progress else 0)]
        if early_stopping:
            callbacks.append(
                lgb.early_stopping(early_stopping, verbose=show_progress)
            )

        print(
            "[Stage 2] Training LightGBM on %d features (%d transformer + %d GP)…"
            % (X_train.shape[1], self.d_model, X_train.shape[1] - self.d_model)
        )

        booster = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        print(
            "[Stage 2] Best LightGBM iteration: %d  val logloss: %.5f"
            % (
                booster.best_iteration,
                booster.best_score["valid_0"]["multi_logloss"],
            )
        )
        return booster

    # ---------------------------------------------------------------------- #
    # Public API                                                             #
    # ---------------------------------------------------------------------- #

    def train(
        self,
        dataset,
        num_folds: int | None = None,
        random_state: int | None = None,
        val_fold: int = 0,
        show_progress: bool = True,
        weight_decay: float = 1e-2,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 2,
        min_lr: float = 1e-6,
        early_stopping_patience: int = 5,
        seq_raw=None,
        gp_raw=None,
    ):
        """Train the two-stage classifier.

        Parameters
        ----------
        dataset
            Avocado dataset. Must be loaded with ``metadata_only=False`` if
            sequence features need to be extracted on the fly.
        num_folds, random_state, val_fold
            Fold-based train/val split, consistent with other classifiers.
        show_progress
            Show tqdm bars and epoch logs.
        weight_decay, lr_scheduler_factor, lr_scheduler_patience, min_lr,
        early_stopping_patience
            Stage-1 transformer training hyper-parameters.
        seq_raw
            Optional precomputed sequence raw features. When provided, the
            sequence branch will use these directly and will not re-extract
            sequence raw features from the dataset.
        gp_raw
            Optional precomputed GP raw features. When provided, the GP branch
            will use these directly and will not read or recompute
            dataset.raw_features.
        """
        self._ensure_torch()
        self._ensure_lgbm()

        if num_folds is None:
            num_folds = settings["num_folds"]
        if random_state is None:
            random_state = settings["fold_random_state"]
        if num_folds < 2:
            raise AvocadoException("TransformerGBMClassifier requires num_folds >= 2.")

        # ---- parse inputs ------------------------------------------------- #
        # seq_raw can be passed in directly to avoid re-extraction.
        # If not provided, _get_seq_raw_features will find it in
        # dataset.raw_features (if pre-extracted) or extract on the fly.
        if seq_raw is None:
            seq_raw = self._get_seq_raw_features(dataset)
        cont_features, band_ids, mask = self._parse_sequence_features(
            dataset, seq_raw=seq_raw
        )

        # GP features can be passed explicitly through gp_raw to avoid
        # overloading dataset.raw_features.
        gp_df = self._get_gp_feature_frame(dataset, gp_raw=gp_raw)

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

        # Fit GP preprocessor on training split only (no leakage)
        self._fit_gp_preprocessor(gp_df.loc[train_mask])
        gp_scaled = self._transform_gp_features(gp_df)

        # ---- Stage 1: train transformer ---------------------------------- #
        encoder = self._train_transformer(
            cont_features=cont_features,
            band_ids=band_ids,
            mask=mask,
            class_indices=class_indices,
            class_names=class_names,
            train_mask=train_mask,
            val_mask=val_mask,
            show_progress=show_progress,
            weight_decay=weight_decay,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            min_lr=min_lr,
            early_stopping_patience=early_stopping_patience,
        )

        # ---- Stage 2: extract embeddings + train LightGBM ---------------- #
        print("[Stage 2] Extracting transformer embeddings…")
        embeddings = self._extract_embeddings(encoder, cont_features, band_ids, mask)

        X = self._build_lgbm_features(embeddings, gp_scaled)

        gbm = self._train_lgbm(
            X_train=X[train_mask],
            y_train=class_indices[train_mask],
            X_val=X[val_mask],
            y_val=class_indices[val_mask],
            class_names=class_names,
            show_progress=show_progress,
        )

        self.encoder = encoder
        self.gbm = gbm
        self.class_names = class_names
        self.device = next(encoder.parameters()).device

        return gbm

    def predict(
        self,
        dataset,
        show_progress: bool = True,
        seq_raw=None,
        gp_raw=None,
    ) -> pd.DataFrame:
        """Generate class probability predictions for a dataset.

        Parameters
        ----------
        dataset
            Avocado dataset.
        show_progress
            Present for API symmetry; currently unused.
        seq_raw
            Optional precomputed sequence raw features. When provided, the
            sequence branch will use these directly.
        gp_raw
            Optional precomputed GP raw features. When provided, the GP branch
            will use these directly.

        Notes
        -----
        If ``seq_raw`` is not provided, the dataset must be loaded with
        ``metadata_only=False`` so that ``dataset.objects`` is available for
        the sequence branch.
        """
        self._ensure_torch()
        self._ensure_lgbm()

        if self.encoder is None or self.gbm is None:
            raise AvocadoException("Model has not been trained yet.")

        if seq_raw is None:
            seq_raw = self._get_seq_raw_features(dataset)

        cont_features, band_ids, mask = self._parse_sequence_features(
            dataset, seq_raw=seq_raw
        )
        gp_df = self._get_gp_feature_frame(dataset, gp_raw=gp_raw)
        gp_scaled = self._transform_gp_features(gp_df)

        embeddings = self._extract_embeddings(
            self.encoder, cont_features, band_ids, mask
        )
        X = self._build_lgbm_features(embeddings, gp_scaled)

        probs = self.gbm.predict(X)          # (n, num_classes)
        probs = probs / probs.sum(axis=1, keepdims=True)

        predictions = pd.DataFrame(
            probs,
            index=dataset.metadata.index,
            columns=self.class_names,
        )
        predictions.index.name = "object_id"
        return predictions