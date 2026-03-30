from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from avocado.classifier import Classifier
from avocado.utils import AvocadoException


class CNNClassifier(Classifier):
    """Simple CNN-based classifier using PyTorch.

    Notes
    -----
    Expects `dataset.select_features(featurizer)` to return a 3D array-like
    with shape `(n_samples, seq_len, channels)`.
    """

    def __init__(
        self,
        name,
        featurizer,
        num_epochs=10,
        batch_size=64,
        lr=1e-3,
        class_weights=None,
        device=None,
    ):
        super().__init__(name)

        self.featurizer = featurizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.class_weights = class_weights
        self.device = device

        self.model = None
        self.class_names = None

    def _build_model(self, seq_len, channels, num_classes):
        try:
            import torch.nn as nn
        except Exception as exc:
            raise AvocadoException(
                "PyTorch is required for CNNClassifier but is not installed."
            ) from exc

        # Inputs are expected in (batch, channels, seq_len) order.
        return nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def _parse_features(self, dataset):
        features = dataset.select_features(self.featurizer)

        if hasattr(features, "values"):
            features = features.values

        features = np.asarray(features, dtype=float)

        if np.isnan(features).any() or np.isinf(features).any():
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if features.ndim != 3:
            raise AvocadoException(
                "CNNClassifier expects features shaped (n, seq_len, channels)."
            )

        return features.astype(np.float32)

    def train(self, dataset, show_progress=True):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except Exception as exc:
            raise AvocadoException(
                "PyTorch is required for CNNClassifier but is not installed."
            ) from exc

        features = self._parse_features(dataset)

        object_classes = dataset.metadata["class"].values
        class_names = np.unique(object_classes)
        class_map = {c: i for i, c in enumerate(class_names)}
        class_indices = np.array([class_map[c] for c in object_classes], dtype=np.int64)

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        X = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(class_indices, dtype=torch.long)

        dataset_torch = TensorDataset(X, y)
        loader = DataLoader(dataset_torch, batch_size=self.batch_size, shuffle=True)

        seq_len = features.shape[1]
        channels = features.shape[2]
        num_classes = len(class_names)

        model = self._build_model(seq_len, channels, num_classes).to(device)

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

        model.train()
        for epoch in iterator:
            epoch_loss = 0.0

            for xb, yb in loader:
                # Convert from (batch, seq_len, channels) to (batch, channels, seq_len)
                xb = xb.permute(0, 2, 1).to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = loss_fn(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(dataset_torch)

            if show_progress:
                tqdm.write("Epoch %d loss: %.5f" % (epoch + 1, epoch_loss))
            else:
                print("Epoch %d loss: %.5f" % (epoch + 1, epoch_loss))

        self.model = model
        self.class_names = class_names
        self.device = device

        return model

    def predict(self, dataset, show_progress=True):
        try:
            import torch
            import torch.nn.functional as F
        except Exception as exc:
            raise AvocadoException(
                "PyTorch is required for CNNClassifier but is not installed."
            ) from exc

        if self.model is None:
            raise AvocadoException("Model has not been trained yet.")

        features = self._parse_features(dataset)

        X = torch.tensor(features, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.permute(0, 2, 1))
            probs = F.softmax(logits, dim=1).cpu().numpy()

        predictions = pd.DataFrame(
            probs,
            index=dataset.metadata.index,
            columns=self.class_names,
        )
        predictions.index.name = "object_id"

        return predictions