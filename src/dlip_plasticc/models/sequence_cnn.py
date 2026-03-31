from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from avocado.classifier import Classifier
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

    def train(self, dataset, show_progress=True):
        self._ensure_torch()

        from torch.utils.data import DataLoader, TensorDataset

        features, mask = self._parse_features(dataset)

        object_classes = dataset.metadata["class"].values
        class_names = np.unique(object_classes)
        class_map = {c: i for i, c in enumerate(class_names)}
        class_indices = np.array([class_map[c] for c in object_classes], dtype=np.int64)

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        X = torch.tensor(features, dtype=torch.float32)
        M = torch.tensor(mask, dtype=torch.float32)
        y = torch.tensor(class_indices, dtype=torch.long)

        ds = TensorDataset(X, M, y)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

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

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        iterator = range(self.num_epochs)
        if show_progress:
            iterator = tqdm(iterator, desc="Epochs", dynamic_ncols=True)

        history = []

        model.train()
        for epoch in iterator:
            epoch_loss = 0.0

            for xb, mb, yb in loader:
                xb = xb.to(device)
                mb = mb.to(device)
                yb = yb.to(device)

                logits = model(xb, mb)
                loss = loss_fn(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(ds)
            history.append({"epoch": epoch + 1, "train_loss": epoch_loss})

            msg = f"Epoch {epoch + 1} loss: {epoch_loss:.5f}"
            if show_progress:
                tqdm.write(msg)
            else:
                print(msg)

        self.model = model
        self.class_names = class_names
        self.device = device
        self.history = pd.DataFrame(history)

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