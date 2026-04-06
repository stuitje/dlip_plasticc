from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from avocado.classifier import Classifier
from avocado.settings import settings
from avocado.utils import AvocadoException


class CNNClassifier(Classifier):
    """CNN-based classifier using PyTorch.

    Features
    --------
    - Fold-based train/val split with early stopping and best-checkpoint restore.
    - ReduceLROnPlateau learning-rate scheduler.
    - Training history stored as a DataFrame in ``self.history``.
    - ``auto_class_weights``: inverse-frequency class weighting computed from
      the training split (approximates Boone's flat-weighted objective).
    - Explicit ``class_weights`` dict for manual per-class multipliers; can be
      combined multiplicatively with ``auto_class_weights``.
    - Supports training/prediction from:
        1) dataset-extracted features,
        2) precomputed reshaped 3D CNN features,
        3) precomputed raw feature tables loaded from disk.

    Notes
    -----
    Expects CNN-ready features shaped ``(n_samples, seq_len, channels)``.

    If you pass ``raw_features=...``, the classifier will call
    ``self.featurizer.select_features(raw_features)`` to convert them into
    CNN-ready 3D features. This is useful when loading saved HDF5 feature
    tables such as ``features_plasticc_augment.h5``.
    """

    def __init__(
        self,
        name,
        featurizer,
        num_epochs=40,
        batch_size=64,
        lr=1e-3,
        class_weights=None,
        auto_class_weights=False,
        device=None,
    ):
        super().__init__(name)

        self.featurizer = featurizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.class_weights = class_weights
        self.auto_class_weights = auto_class_weights
        self.device = device

        self.model = None
        self.class_names = None
        self.history = None
        self.best_val_loss: float = np.inf
        self.val_fold = None

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                       #
    # ---------------------------------------------------------------------- #

    def _build_model(self, seq_len, channels, num_classes):
        try:
            import torch.nn as nn
        except Exception as exc:
            raise AvocadoException(
                "PyTorch is required for CNNClassifier but is not installed."
            ) from exc

        # Input convention: (batch, channels, seq_len)
        return nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def _parse_features(self, dataset):
        """Load CNN-ready features from an Avocado dataset."""
        features = dataset.select_features(self.featurizer)
        return self._coerce_3d_features(features, source="dataset.select_features")

    def _coerce_3d_features(self, features, source="features"):
        """Validate/sanitise CNN features and ensure shape (n, seq_len, channels)."""
        if hasattr(features, "values"):
            features = features.values

        features = np.asarray(features, dtype=float)

        if np.isnan(features).any() or np.isinf(features).any():
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if features.ndim != 3:
            raise AvocadoException(
                "%s must be shaped (n_samples, seq_len, channels); "
                "got array with shape %s." % (source, features.shape)
            )

        return features.astype(np.float32)

    def _prepare_features(self, dataset=None, features=None, raw_features=None):
        """Prepare CNN-ready features from one of three sources.

        Parameters
        ----------
        dataset
            Avocado dataset. Used when ``features`` and ``raw_features`` are not
            provided, or for row alignment / length checks.
        features
            Precomputed CNN-ready array of shape (n, seq_len, channels).
        raw_features
            Precomputed raw feature table, e.g. loaded from HDF5. This will be
            passed through ``self.featurizer.select_features(raw_features)``.
        """
        if features is not None and raw_features is not None:
            raise AvocadoException(
                "Pass only one of 'features' or 'raw_features', not both."
            )

        if features is not None:
            prepared = self._coerce_3d_features(features, source="features")

        elif raw_features is not None:
            aligned_raw_features = raw_features

            # If possible, align row order to dataset.metadata.index
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

            prepared = self._coerce_3d_features(
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
            inv_freq /= inv_freq.mean()  # normalise: mean weight = 1
            weights *= inv_freq

        if self.class_weights is not None:
            for i, c in enumerate(class_names):
                weights[i] *= self.class_weights.get(c, 1.0)

        return torch.tensor(weights, dtype=torch.float32).to(device)

    # ---------------------------------------------------------------------- #
    # Public API                                                             #
    # ---------------------------------------------------------------------- #

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
        early_stopping_patience=7,
        features=None,
        raw_features=None,
    ):
        """Train the CNN with fold-based validation, LR scheduling and early stopping.

        Parameters
        ----------
        dataset
            Avocado dataset with metadata available.
        num_folds, random_state, val_fold
            Fold-based train/val split.
        show_progress
            Show tqdm epoch bar and per-epoch logs.
        weight_decay
            L2 regularisation for AdamW.
        lr_scheduler_factor, lr_scheduler_patience, min_lr
            ReduceLROnPlateau settings.
        early_stopping_patience
            Stop training if val loss does not improve for this many epochs.
        features
            Optional precomputed CNN-ready features shaped
            ``(n_samples, seq_len, channels)``.
        raw_features
            Optional precomputed raw feature table. If provided, this is passed
            through ``self.featurizer.select_features(raw_features)``.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as exc:
            raise AvocadoException(
                "PyTorch is required for CNNClassifier but is not installed."
            ) from exc

        if num_folds is None:
            num_folds = settings["num_folds"]
        if random_state is None:
            random_state = settings["fold_random_state"]

        if num_folds < 2:
            raise AvocadoException("CNNClassifier requires num_folds >= 2.")

        features = self._prepare_features(
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

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # Tensors — permute to (batch, channels, seq_len) for Conv1d
        def to_tensor(arr, idx):
            return torch.tensor(arr[idx], dtype=torch.float32).permute(0, 2, 1)

        X_train = to_tensor(features, train_mask)
        X_val = to_tensor(features, val_mask)
        y_train = torch.tensor(class_indices[train_mask], dtype=torch.long)
        y_val = torch.tensor(class_indices[val_mask], dtype=torch.long)

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )

        channels = features.shape[2]
        seq_len = features.shape[1]
        num_classes = len(class_names)

        model = self._build_model(seq_len, channels, num_classes).to(device)

        weight_tensor = self._build_weight_tensor(
            class_names,
            class_indices[train_mask],
            device,
        )
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

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

        best_val_loss = np.inf
        best_state = None
        history = []
        no_improve = 0

        iterator = range(self.num_epochs)
        if show_progress:
            iterator = tqdm(iterator, desc="Epochs", dynamic_ncols=True)

        for epoch in iterator:
            # train
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * xb.size(0)

            train_loss /= len(train_ds)

            # validate 
            model.eval()
            val_loss = 0.0
            correct = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    val_loss += loss_fn(logits, yb).item() * xb.size(0)
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
                    for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

            if early_stopping_patience and no_improve >= early_stopping_patience:
                msg = (
                    "Early stopping at epoch %d.  Best val=%.5f"
                    % (epoch + 1, best_val_loss)
                )
                tqdm.write(msg) if show_progress else print(msg)
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.model = model
        self.class_names = class_names
        self.device = device
        self.history = pd.DataFrame(history)
        self.best_val_loss = float(best_val_loss)
        self.val_fold = val_fold

        print("Best validation loss: %.5f" % best_val_loss)
        return model

    def predict(
        self,
        dataset,
        show_progress=True,
        features=None,
        raw_features=None,
    ):
        """Predict class probabilities.

        Parameters
        ----------
        dataset
            Avocado dataset with metadata available.
        show_progress
            Whether to show a tqdm progress bar during batched prediction.
        features
            Optional precomputed CNN-ready features shaped
            ``(n_samples, seq_len, channels)``.
        raw_features
            Optional precomputed raw feature table. If provided, this is passed
            through ``self.featurizer.select_features(raw_features)``.
        """
        try:
            import torch
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as exc:
            raise AvocadoException(
                "PyTorch is required for CNNClassifier but is not installed."
            ) from exc

        if self.model is None:
            raise AvocadoException("Model has not been trained yet.")

        features = self._prepare_features(
            dataset=dataset,
            features=features,
            raw_features=raw_features,
        )

        X = torch.tensor(features, dtype=torch.float32).permute(0, 2, 1)
        pred_ds = TensorDataset(X)
        pred_loader = DataLoader(
            pred_ds,
            batch_size=self.batch_size,
            shuffle=False,
        )

        iterator = pred_loader
        if show_progress:
            iterator = tqdm(iterator, desc="Predict", dynamic_ncols=True)

        probs_chunks = []

        self.model.eval()
        with torch.no_grad():
            for (xb,) in iterator:
                xb = xb.to(self.device)
                logits = self.model(xb)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                probs_chunks.append(probs)

        probs = np.concatenate(probs_chunks, axis=0)

        predictions = pd.DataFrame(
            probs,
            index=dataset.metadata.index,
            columns=self.class_names,
        )
        predictions.index.name = "object_id"
        return predictions