import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn

from avocado.settings import settings
from avocado.utils import AvocadoException
from avocado.classifier import Classifier

class CNNClassifier(Classifier):
    """Simple CNN-based classifier using PyTorch.

    Notes
    -----
    - This classifier expects `dataset.select_features(featurizer)` to return
      a 3D numpy array or array-like with shape `(n_samples, seq_len, channels)`.
    - PyTorch is an optional dependency; an informative error is raised if not
      installed.
    - `class_weights` (optional) may be provided as a dict mapping class name
      to a scalar; these are converted to loss weights.
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

        # These will be set while training
        self.model = None
        self.class_names = None

    def _build_model(self, seq_len, channels, num_classes):
        # Build a sequential model (picklable) and expect inputs in
        # (batch, channels, seq_len) order. The caller permutes inputs.
        import torch.nn as nn

        model = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

        return model

    def train(
        self,
        dataset,
        show_progress=True,
    ):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except Exception:
            raise AvocadoException(
                "PyTorch is required for CNNClassifier but is not installed."
            )

        features = dataset.select_features(self.featurizer)

        # Accept pandas DataFrame/Series or numpy arrays
        if hasattr(features, "values"):
            features = features.values

        features = np.asarray(features, dtype=float)

        # Sanitize NaN / Inf values which cause training to diverge
        if np.isnan(features).any() or np.isinf(features).any():
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if features.ndim != 3:
            raise AvocadoException(
                "CNNClassifier expects features shaped (n, seq_len, channels)."
            )

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
                # xb from DataLoader is (batch, seq_len, channels); convert
                # to (batch, channels, seq_len) for Conv1d.
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

        self.model = model
        self.class_names = class_names
        self.device = device

        return model

    def predict(self, dataset, show_progress=True):
        try:
            import torch
            import torch.nn.functional as F
        except Exception:
            raise AvocadoException(
                "PyTorch is required for CNNClassifier but is not installed."
            )

        if self.model is None:
            raise AvocadoException("Model has not been trained yet.")

        features = dataset.select_features(self.featurizer)
        if hasattr(features, "values"):
            features = features.values
        features = np.asarray(features, dtype=float)

        # Sanitize
        if np.isnan(features).any() or np.isinf(features).any():
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if features.ndim != 3:
            raise AvocadoException(
                "CNNClassifier expects features shaped (n, seq_len, channels)."
            )

        X = torch.tensor(features, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            # permute to (batch, channels, seq_len)
            logits = self.model(X.permute(0, 2, 1))
            probs = F.softmax(logits, dim=1).cpu().numpy()

        predictions = pd.DataFrame(
            probs,
            index=dataset.metadata.index,   # <-- this is the crucial fix
            columns=self.class_names,
        )
        predictions.index.name = "object_id"  # <-- also crucial

        return predictions


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class _TransformerNet(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
        max_len=512,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.pos_enc(x)

        pad_mask = None
        if mask is not None:
            pad_mask = (mask == 0)

        x = self.encoder(x, src_key_padding_mask=pad_mask)

        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        else:
            x = x.mean(dim=1)

        return self.head(x)


class TransformerClassifier(Classifier):
    """Transformer-based classifier using PyTorch.

    Expects dataset.select_features(featurizer) to return either:
      - features with shape (n_samples, seq_len, channels), or
      - (features, mask), where mask has shape (n_samples, seq_len)
        and uses 1 for valid timesteps, 0 for padding.
    """

    def __init__(
        self,
        name,
        featurizer,
        num_epochs=15,
        batch_size=64,
        lr=1e-4,
        class_weights=None,
        device=None,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
    ):
        super().__init__(name)

        self.featurizer = featurizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.class_weights = class_weights
        self.device = device

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.model = None
        self.class_names = None
        self.best_val_loss = None
        self.history = None
        self.val_fold = None

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
                "TransformerClassifier expects features shaped (n, seq_len, channels), "
                "or (features, mask)."
            )

        if mask is None:
            mask = np.ones(features.shape[:2], dtype=np.float32)
        else:
            if hasattr(mask, "values"):
                mask = mask.values
            mask = np.asarray(mask, dtype=np.float32)

        return features, mask

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
        try:
            from torch.utils.data import DataLoader, TensorDataset
        except Exception:
            raise AvocadoException(
                "PyTorch is required for TransformerClassifier but is not installed."
            )

        if num_folds is None:
            num_folds = settings["num_folds"]
        if random_state is None:
            random_state = settings["fold_random_state"]

        if num_folds < 2:
            raise AvocadoException("TransformerClassifier requires num_folds >= 2.")

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

        model = _TransformerNet(
            input_dim=features.shape[2],
            num_classes=len(class_names),
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_len=features.shape[1],
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
        self.best_val_loss = best_val_loss
        self.history = pd.DataFrame(history)
        self.val_fold = val_fold

        print("Best validation loss: %.5f" % best_val_loss)

        return model

    def predict(self, dataset, show_progress=True):
        try:
            import torch.nn.functional as F
        except Exception:
            raise AvocadoException(
                "PyTorch is required for TransformerClassifier but is not installed."
            )

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