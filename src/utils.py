import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
import avocado

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH    = "/scratch/s4339150/plasticc/features/features_v1_plasticc_train.h5"
NUM_CHUNKS       = 8
NUM_TRAIN_CHUNKS = 5
DROPOUT          = 0.4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        log_p   = F.log_softmax(logits, dim=1)
        p       = torch.exp(log_p)
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t     = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        loss = -focal_weight * log_p_t
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()

# ── Model ─────────────────────────────────────────────────────────────────────
class PlasticcNet(nn.Module):
    def __init__(self, in_dim, n_classes, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)

# ── Data loading & preprocessing ─────────────────────────────────────────────
def load_features():
    """Load raw features from HDF5."""
    df = pd.read_hdf(FEATURES_PATH, key="raw_features")
    return df

def load_labels():
    """Load class labels for the featurized chunks."""
    labels = []
    for chunk in range(NUM_TRAIN_CHUNKS):
        d = avocado.load("plasticc_train", chunk=chunk,
                         num_chunks=NUM_CHUNKS, metadata_only=True)
        labels.append(d.metadata[["class"]])
    return pd.concat(labels)

def preprocess(df):
    """Add missingness indicators and median-impute. Returns numpy array."""
    nan_cols = df.columns[df.isnull().any()]
    indicators = pd.DataFrame(
        {f"{col}_missing": df[col].isnull().astype(float) for col in nan_cols},
        index=df.index
    )
    df = pd.concat([df, indicators], axis=1)
    df = df.fillna(df.median())
    assert df.isnull().sum().sum() == 0, "NaNs remain after imputation"
    return df

def build_dataset():
    """
    Full pipeline: load → preprocess → encode → scale.
    Returns X (np.float32), y (np.int64), label_encoder, scaler.
    """
    print("Loading features...")
    df = load_features()
    print(f"  Raw shape: {df.shape}")

    print("Loading labels...")
    labels = load_labels().loc[df.index]
    print(f"  Classes:\n{labels['class'].value_counts()}")

    print("Preprocessing...")
    df = preprocess(df)
    print(f"  Final shape: {df.shape}")

    le = LabelEncoder()
    y  = le.fit_transform(labels["class"].values).astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(df.values).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, le, scaler

def class_weights(y, n_classes, device):
    """Inverse-frequency class weights, normalized."""
    counts = np.bincount(y)
    w = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
    return w / w.sum() * n_classes