from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from avocado.classifier import Classifier
from avocado.settings import settings
from avocado.utils import AvocadoException

# Module-level model classes (required for pickle/classifier.write)
try:
    import torch
    import torch.nn as nn

    class ResidualBlock(nn.Module):
        def __init__(self, in_dim, out_dim, dropout=0.3, use_batch_norm=True):
            super().__init__()
            self.lin1 = nn.Linear(in_dim, out_dim)
            self.bn1 = nn.BatchNorm1d(out_dim) if use_batch_norm else nn.Identity()
            self.lin2 = nn.Linear(out_dim, out_dim)
            self.bn2 = nn.BatchNorm1d(out_dim) if use_batch_norm else nn.Identity()
            self.act = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        def forward(self, x):
            residual = self.skip(x)
            x = self.lin1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.lin2(x)
            x = self.bn2(x)
            x = x + residual
            x = self.act(x)
            x = self.dropout(x)
            return x

    class ResidualMLP(nn.Module):
        def __init__(self, input_dim, hidden_dims, num_classes, dropout, use_batch_norm):
            super().__init__()
            dims = [input_dim] + list(hidden_dims)
            self.blocks = nn.ModuleList(
                [
                    ResidualBlock(
                        dims[i],
                        dims[i + 1],
                        dropout=dropout,
                        use_batch_norm=use_batch_norm,
                    )
                    for i in range(len(dims) - 1)
                ]
            )
            self.head = nn.Linear(hidden_dims[-1], num_classes)

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.head(x)

except ImportError:
    pass


class MLPClassifier(Classifier):
    """Residual MLP classifier using PyTorch.

    Features
    --------
    - Fold-based train/val split with early stopping and best-checkpoint restore.
    - ReduceLROnPlateau learning-rate scheduler.
    - Training history stored as a DataFrame in ``self.history``.
    - ``auto_class_weights``: inverse-frequency class weighting computed from
      the training split.
    - Explicit ``class_weights`` dict for manual per-class multipliers.
    - Supports training/prediction from:
        1) dataset-extracted features,
        2) precomputed 2D feature arrays,
        3) precomputed raw feature tables loaded from disk.
    - Optional standardisation using training-split statistics only.
    - Optional label smoothing.
    - Optional per-epoch validation flat-weighted log-loss tracking.

    Notes
    -----
    Expects flat tabular features shaped ``(n_samples, n_features)``.
    If 3D features are passed accidentally, they are flattened to 2D.
    """

    def __init__(
        self,
        name,
        featurizer,
        hidden_dims=(512, 256, 128),
        num_epochs=60,
        batch_size=256,
        lr=1e-3,
        dropout=0.3,
        class_weights=None,
        auto_class_weights=False,
        device=None,
        use_batch_norm=True,
        standardize=True,
        label_smoothing=0.05,
        track_flat_score=True,
        flat_class_weights=None,
        metric_clip_eps=1e-15,
    ):
        super().__init__(name)

        self.featurizer = featurizer
        self.hidden_dims = tuple(hidden_dims)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.class_weights = class_weights
        self.auto_class_weights = auto_class_weights
        self.device = device
        self.use_batch_norm = use_batch_norm
        self.standardize = standardize
        self.label_smoothing = label_smoothing
        self.track_flat_score = track_flat_score
        self.flat_class_weights = flat_class_weights
        self.metric_clip_eps = metric_clip_eps

        self.model = None
        self.class_names = None
        self.history = None
        self.best_val_loss: float = np.inf
        self.val_fold = None
        self.input_dim = None

        self.feature_mean_ = None
        self.feature_std_ = None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_model(self, input_dim, num_classes):
        try:
            import torch.nn as nn
        except Exception as exc:
            raise AvocadoException(
                "PyTorch is required for MLPClassifier but is not installed."
            ) from exc

        if len(self.hidden_dims) == 0:
            raise AvocadoException("hidden_dims must contain at least one layer.")

        return ResidualMLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            num_classes=num_classes,
            dropout=self.dropout,
            use_batch_norm=self.use_batch_norm,
        )

    def _coerce_2d_features(self, features, source="features"):
        """Validate/sanitise flat features and ensure shape (n, f)."""
        if hasattr(features, "values"):
            features = features.values

        features = np.asarray(features, dtype=float)

        if np.isnan(features).any() or np.isinf(features).any():
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if features.ndim == 3:
            features = features.reshape(features.shape[0], -1)

        if features.ndim != 2:
            raise AvocadoException(
                "%s must be shaped (n_samples, n_features); got array with "
                "shape %s." % (source, features.shape)
            )

        return features.astype(np.float32)

    def _parse_features(self, dataset):
        """Load flat features from an Avocado dataset."""
        features = dataset.select_features(self.featurizer)
        return self._coerce_2d_features(features, source="dataset.select_features")

    def _prepare_features(self, dataset=None, features=None, raw_features=None):
        """Prepare flat features from one of three sources."""
        if features is not None and raw_features is not None:
            raise AvocadoException(
                "Pass only one of 'features' or 'raw_features', not both."
            )

        if features is not None:
            prepared = self._coerce_2d_features(features, source="features")

        elif raw_features is not None:
            aligned_raw_features = raw_features

            if dataset is not None and hasattr(aligned_raw_features, "loc"):
                if not hasattr(dataset, "metadata"):
                    raise AvocadoException(
                        "Dataset must provide metadata when using raw_features."
                    )
                try:
                    aligned_raw_features = aligned_raw_features.loc[
                        dataset.metadata.index
                    ]
                except Exception as exc:
                    raise AvocadoException(
                        "Failed to align raw_features to dataset.metadata.index. "
                        "Make sure raw_features is indexed by object_id and "
                        "contains the same rows as the dataset."
                    ) from exc

            try:
                prepared = self.featurizer.select_features(aligned_raw_features)
            except Exception as exc:
                raise AvocadoException(
                    "Failed to convert raw_features with featurizer.select_features."
                ) from exc

            prepared = self._coerce_2d_features(
                prepared,
                source="featurizer.select_features(raw_features)",
            )

        else:
            if dataset is None:
                raise AvocadoException(
                    "A dataset is required when 'features' and 'raw_features' "
                    "are not provided."
                )
            prepared = self._parse_features(dataset)

        if dataset is not None:
            expected_n = len(dataset.metadata)
            actual_n = len(prepared)
            if actual_n != expected_n:
                raise AvocadoException(
                    "Number of feature rows (%d) does not match dataset size (%d)."
                    % (actual_n, expected_n)
                )

        return prepared

    def _fit_standardizer(self, X_train):
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        self.feature_mean_ = mean.astype(np.float32)
        self.feature_std_ = std.astype(np.float32)

    def _apply_standardizer(self, X):
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise AvocadoException(
                "Standardizer has not been fitted. Train the model first."
            )
        return ((X - self.feature_mean_) / self.feature_std_).astype(np.float32)

    def _build_weight_tensor(self, class_names, class_indices_train, device):
        """Return a 1-D weight tensor for CrossEntropyLoss, or None."""
        import torch

        if not self.auto_class_weights and self.class_weights is None:
            return None

        weights = np.ones(len(class_names), dtype=np.float32)

        if self.auto_class_weights:
            counts = np.bincount(
                class_indices_train,
                minlength=len(class_names),
            ).astype(np.float32)
            counts = np.where(counts == 0, 1.0, counts)
            inv_freq = 1.0 / counts
            inv_freq /= inv_freq.mean()
            weights *= inv_freq

        if self.class_weights is not None:
            for i, c in enumerate(class_names):
                weights[i] *= self.class_weights.get(c, 1.0)

        return torch.tensor(weights, dtype=torch.float32).to(device)

    def _compute_flat_score(self, true_labels, probs, class_names):
        """Compute PLAsTiCC-style flat-weighted log-loss on a validation fold."""
        import avocado

        pred_df = pd.DataFrame(probs, columns=class_names)

        if self.metric_clip_eps is not None and self.metric_clip_eps > 0:
            pred_df = pred_df.clip(lower=self.metric_clip_eps, upper=1.0)
            pred_df = pred_df.div(pred_df.sum(axis=1), axis=0)

        class_weights = self.flat_class_weights
        if class_weights is None:
            class_weights = avocado.plasticc.plasticc_flat_weights

        return avocado.weighted_multi_logloss(
            np.asarray(true_labels),
            pred_df,
            class_weights=class_weights,
        )

    # Public API                                                           

    def train(
        self,
        dataset,
        num_folds=None,
        random_state=None,
        val_fold=0,
        show_progress=True,
        weight_decay=1e-4,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=3,
        min_lr=1e-6,
        early_stopping_patience=10,
        features=None,
        raw_features=None,
    ):
        """Train the MLP with fold-based validation, LR scheduling and early stopping."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as exc:
            raise AvocadoException(
                "PyTorch is required for MLPClassifier but is not installed."
            ) from exc

        if num_folds is None:
            num_folds = settings["num_folds"]
        if random_state is None:
            random_state = settings["fold_random_state"]

        if num_folds < 2:
            raise AvocadoException("MLPClassifier requires num_folds >= 2.")

        X = self._prepare_features(
            dataset=dataset,
            features=features,
            raw_features=raw_features,
        )

        object_classes = dataset.metadata["class"].values
        class_names = np.unique(object_classes)
        class_map = {c: i for i, c in enumerate(class_names)}
        class_indices = np.array(
            [class_map[c] for c in object_classes],
            dtype=np.int64,
        )

        folds = dataset.label_folds(num_folds, random_state)
        val_mask = folds == val_fold
        train_mask = folds != val_fold

        if np.sum(val_mask) == 0:
            raise AvocadoException("Validation fold is empty.")
        if np.sum(train_mask) == 0:
            raise AvocadoException("Training split is empty.")

        X_train = X[train_mask].copy()
        X_val   = X[val_mask].copy()

        if self.standardize:
            self._fit_standardizer(X_train)
            X_train = self._apply_standardizer(X_train)
            X_val   = self._apply_standardizer(X_val)
        else:
            self.feature_mean_ = None
            self.feature_std_  = None

        y_train_np   = class_indices[train_mask]
        y_val_np     = class_indices[val_mask]
        y_val_labels = object_classes[val_mask]

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
        y_train_t = torch.tensor(y_train_np, dtype=torch.long)
        y_val_t   = torch.tensor(y_val_np,   dtype=torch.long)

        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds   = TensorDataset(X_val_t,   y_val_t)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        input_dim   = X_train.shape[1]
        num_classes = len(class_names)

        model = self._build_model(input_dim, num_classes).to(device)

        weight_tensor = self._build_weight_tensor(class_names, y_train_np, device)
        loss_fn = nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=self.label_smoothing,
        )

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

        best_val_loss = np.inf
        best_state    = None
        history       = []
        no_improve    = 0

        iterator = range(self.num_epochs)
        if show_progress:
            iterator = tqdm(iterator, desc="Epochs", dynamic_ncols=True)

        for epoch in iterator:
            # ---- train ----
            model.train()
            train_loss    = 0.0
            train_correct = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss   = loss_fn(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss    += loss.item() * xb.size(0)
                train_correct += (logits.argmax(1) == yb).sum().item()

            train_loss /= len(train_ds)
            train_acc   = train_correct / len(train_ds)

            # ---- validate ----
            model.eval()
            val_loss         = 0.0
            val_correct      = 0
            val_probs_chunks = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss    += loss_fn(logits := model(xb), yb).item() * xb.size(0)
                    val_correct += (logits.argmax(1) == yb).sum().item()
                    val_probs_chunks.append(
                        torch.softmax(logits, dim=1).cpu().numpy()
                    )

            val_loss /= len(val_ds)
            val_acc   = val_correct / len(val_ds)
            val_probs = np.concatenate(val_probs_chunks, axis=0)

            val_flat_score = np.nan
            if self.track_flat_score:
                try:
                    val_flat_score = self._compute_flat_score(
                        true_labels=y_val_labels,
                        probs=val_probs,
                        class_names=class_names,
                    )
                except Exception:
                    val_flat_score = np.nan

            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]

            history_row = dict(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=lr,
            )
            if self.track_flat_score:
                history_row["val_flat_score"] = val_flat_score
            history.append(history_row)

            msg = (
                "Epoch %d  train=%.5f  train_acc=%.4f  val=%.5f  val_acc=%.4f  lr=%.2e"
                % (epoch + 1, train_loss, train_acc, val_loss, val_acc, lr)
            )
            if self.track_flat_score and np.isfinite(val_flat_score):
                msg += "  flat=%.5f" % val_flat_score
            tqdm.write(msg) if show_progress else print(msg)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

            if early_stopping_patience and no_improve >= early_stopping_patience:
                msg = "Early stopping at epoch %d.  Best val=%.5f" % (
                    epoch + 1, best_val_loss
                )
                tqdm.write(msg) if show_progress else print(msg)
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model        = model
        self.class_names  = class_names
        self.device       = device
        self.history      = pd.DataFrame(history)
        self.best_val_loss = float(best_val_loss)
        self.val_fold     = val_fold
        self.input_dim    = input_dim

        print("Best validation loss: %.5f" % best_val_loss)
        return model

    def predict(
        self,
        dataset,
        show_progress=True,
        features=None,
        raw_features=None,
        clip_eps=1e-15,
    ):
        """Predict class probabilities."""
        try:
            import torch
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as exc:
            raise AvocadoException(
                "PyTorch is required for MLPClassifier but is not installed."
            ) from exc

        if self.model is None:
            raise AvocadoException("Model has not been trained yet.")

        X = self._prepare_features(
            dataset=dataset,
            features=features,
            raw_features=raw_features,
        )

        if self.standardize:
            X = self._apply_standardizer(X)

        X_t      = torch.tensor(X, dtype=torch.float32)
        pred_ds  = TensorDataset(X_t)
        pred_loader = DataLoader(pred_ds, batch_size=self.batch_size, shuffle=False)

        iterator = pred_loader
        if show_progress:
            iterator = tqdm(iterator, desc="Predict", dynamic_ncols=True)

        probs_chunks = []
        self.model.eval()
        with torch.no_grad():
            for (xb,) in iterator:
                xb = xb.to(self.device)
                logits = self.model(xb)
                probs_chunks.append(F.softmax(logits, dim=1).cpu().numpy())

        probs = np.concatenate(probs_chunks, axis=0)

        if clip_eps is not None and clip_eps > 0:
            probs = np.clip(probs, clip_eps, 1.0)
            probs = probs / probs.sum(axis=1, keepdims=True)

        predictions = pd.DataFrame(
            probs,
            index=dataset.metadata.index,
            columns=self.class_names,
        )
        predictions.index.name = "object_id"
        return predictions