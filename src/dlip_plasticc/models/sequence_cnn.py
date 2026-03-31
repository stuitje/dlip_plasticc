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


class _SequenceCNNNet(nn.Module):
    """1D CNN over padded light-curve sequences."""

    def __init__(self, input_channels, num_classes, dropout=0.2):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, mask=None):
        # x: (batch, seq_len, channels)
        x = x.transpose(1, 2)  # -> (batch, channels, seq_len)
        x = self.backbone(x)   # -> (batch, hidden, seq_len)

        if mask is not None:
            mask = mask.unsqueeze(1).float()  # (batch, 1, seq_len)
            x = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
        else:
            x = x.mean(dim=2)

        return self.head(x)


class SequenceCNNClassifier(Classifier):
    """CNN classifier trained on light-curve sequences.

    Expects `dataset.select_features(featurizer)` to return either:
      - features with shape (n_samples, seq_len, channels), or
      - (features, mask), where mask has shape (n_samples, seq_len)
        and uses 1 for valid timesteps and 0 for padding.
    """

    def __init__(
        self,
        name,
        featurizer,
        num_epochs=20,
        batch_size=64,
        lr=1e-3,
        class_weights=None,
        device=None,
        dropout=0.2,
    ):
        super().__init__(name)

        self.featurizer = featurizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.class_weights = class_weights
        self.device = device
        self.dropout = dropout

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

    def _parse_features(self, dataset):
        features = dataset.select_features(self.featurizer)

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
                "SequenceCNNClassifier expects features shaped "
                "(n, seq_len, channels), or (features, mask)."
            )

        if mask is None:
            mask = np.ones(features.shape[:2], dtype=np.float32)
        else:
            if hasattr(mask, "values"):
                mask = mask.values
            mask = np.asarray(mask, dtype=np.float32)

        return features.astype(np.float32), mask.astype(np.float32)

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
                "PyTorch is required for SequenceCNNClassifier but is not installed."
            ) from exc

        if num_folds is None:
            num_folds = settings["num_folds"]
        if random_state is None:
            random_state = settings["fold_random_state"]

        if num_folds < 2:
            raise AvocadoException("SequenceCNNClassifier requires num_folds >= 2.")

        features, mask = self._parse_features(dataset)

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

        X_train = torch.tensor(features[train_mask], dtype=torch.float32)
        M_train = torch.tensor(mask[train_mask], dtype=torch.float32)
        y_train = torch.tensor(class_indices[train_mask], dtype=torch.long)

        X_val = torch.tensor(features[val_mask], dtype=torch.float32)
        M_val = torch.tensor(mask[val_mask], dtype=torch.float32)
        y_val = torch.tensor(class_indices[val_mask], dtype=torch.long)

        train_ds = TensorDataset(X_train, M_train, y_train)
        val_ds = TensorDataset(X_val, M_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = _SequenceCNNNet(
            input_channels=features.shape[2],
            num_classes=len(class_names),
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

            for xb, mb, yb in train_loader:
                xb = xb.to(device)
                mb = mb.to(device)
                yb = yb.to(device)

                logits = model(xb, mb)
                loss = loss_fn(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * xb.size(0)

            train_loss /= len(train_ds)

            # ---- validation ----
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for xb, mb, yb in val_loader:
                    xb = xb.to(device)
                    mb = mb.to(device)
                    yb = yb.to(device)

                    logits = model(xb, mb)
                    loss = loss_fn(logits, yb)
                    val_loss += loss.item() * xb.size(0)

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
        self.history = pd.DataFrame(history)
        self.best_val_loss = best_val_loss
        self.val_fold = val_fold

        print("Best validation loss: %.5f" % best_val_loss)

        return model

    def predict(self, dataset, show_progress=True):
        self._ensure_torch()

        import torch.nn.functional as F

        if self.model is None:
            raise AvocadoException("Model has not been trained yet.")

        features, mask = self._parse_features(dataset)

        X = torch.tensor(features, dtype=torch.float32).to(self.device)
        M = torch.tensor(mask, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X, M)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        predictions = pd.DataFrame(
            probs,
            index=dataset.metadata.index,
            columns=self.class_names,
        )
        predictions.index.name = "object_id"

        return predictions