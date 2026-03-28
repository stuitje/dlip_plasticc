import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
import avocado
import math

from config import (
    FEATURES_PATH, NUM_CHUNKS, NUM_TRAIN_CHUNKS,
    DROPOUT, SEQ_LEN, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, DEVICE
)

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


# ── Redshift-weighted Loss ────────────────────────────────────────────────────
class RedshiftWeightedLoss(nn.Module):
    """
    Weights each object's loss by 1/p(z|class) to correct for the
    spectroscopic bias in the training set.

    Objects at high redshift are underrepresented in training but
    common in the test set — upweighting them corrects for this.

    Following Boone (2019): weight = 1 / (count of objects in redshift bin
    for that class), normalized per class.

    In practice we use a simpler proxy: weight by redshift directly,
    so high-z objects get more weight. This is a reasonable approximation
    when the training set is heavily biased toward low-z objects.
    """
    def __init__(self, gamma=2.0, class_weight=None, redshift_scale=1.0):
        super().__init__()
        self.focal     = FocalLoss(gamma=gamma, weight=class_weight)
        self.redshift_scale = redshift_scale  # tune this — higher = more z weighting

    def forward(self, logits, targets, redshifts=None):
        # Per-sample focal loss (unreduced)
        log_p   = F.log_softmax(logits, dim=1)
        p       = torch.exp(log_p)
        log_p_t = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t     = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.focal.gamma
        loss = -focal_weight * log_p_t

        if self.focal.weight is not None:
            loss = loss * self.focal.weight[targets]

        if redshifts is not None:
            # Upweight high-redshift objects
            # w(z) = 1 + redshift_scale * z  (linear upweighting)
            z_weight = 1.0 + self.redshift_scale * redshifts.to(loss.device)
            z_weight = z_weight / z_weight.mean()  # normalize so mean weight = 1
            loss = loss * z_weight

        return loss.mean()


# ── Time-based Positional Encoding ───────────────────────────────────────────
class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.d_model = d_model
        i = torch.arange(0, d_model // 2, dtype=torch.float32)
        self.register_buffer("div_term", torch.exp(i * (-math.log(10000.0) / d_model)))

    def forward(self, times):
        t       = times.unsqueeze(-1) * 10000.0
        sin_enc = torch.sin(t * self.div_term)
        cos_enc = torch.cos(t * self.div_term)
        return torch.cat([sin_enc, cos_enc], dim=-1)


# ── Transformer branch ────────────────────────────────────────────────────────
class LCTransformer(nn.Module):
    def __init__(self, in_features=5, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.time_pe    = TimePositionalEncoding(d_model)
        self.dropout    = nn.Dropout(dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, mask):
        times    = x[:, :, 0]
        x_proj   = self.input_proj(x)
        pe       = self.time_pe(times)
        x_proj   = self.dropout(x_proj + pe)
        pad_mask = (mask == 0)
        x_proj   = self.encoder(x_proj, src_key_padding_mask=pad_mask)
        mask_exp = mask.unsqueeze(-1)
        return (x_proj * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)


# ── Feature MLP branch ────────────────────────────────────────────────────────
class FeatureMLP(nn.Module):
    def __init__(self, in_dim, d_model=D_MODEL, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_model),
        )
    def forward(self, x):
        return self.net(x)


# ── Hybrid model ──────────────────────────────────────────────────────────────
class HybridPlasticcNet(nn.Module):
    def __init__(self, feature_dim, n_classes, dropout=DROPOUT):
        super().__init__()
        self.transformer = LCTransformer(dropout=dropout)
        self.feature_mlp = FeatureMLP(feature_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL * 2),
            nn.Linear(D_MODEL * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, seq, mask, features):
        lc_emb   = self.transformer(seq, mask)
        feat_emb = self.feature_mlp(features)
        return self.head(torch.cat([lc_emb, feat_emb], dim=1))


# ── Keep old PlasticcNet for backwards compat ─────────────────────────────────
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
    return pd.read_hdf(FEATURES_PATH, key="raw_features")

def load_labels():
    labels = []
    for chunk in range(NUM_TRAIN_CHUNKS):
        d = avocado.load("plasticc_augmented", chunk=chunk,
                         num_chunks=NUM_CHUNKS, metadata_only=True)
        labels.append(d.metadata[["class"]])
    return pd.concat(labels)

def preprocess(df):
    nan_cols = df.columns[df.isnull().any()]
    indicators = pd.DataFrame(
        {f"{col}_missing": df[col].isnull().astype(float) for col in nan_cols},
        index=df.index
    )
    df = pd.concat([df, indicators], axis=1)
    df = df.fillna(df.median())
    assert df.isnull().sum().sum() == 0
    return df

def build_dataset():
    print("Loading features...")
    df = load_features()
    print(f"  Raw shape: {df.shape}")
    print("Loading labels...")
    labels = load_labels().loc[df.index]
    print("Preprocessing...")
    df = preprocess(df)
    print(f"  Final shape: {df.shape}")
    le = LabelEncoder()
    y  = le.fit_transform(labels["class"].values).astype(np.int64)
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, le, scaler, df.index.tolist()

def class_weights(y, n_classes, device):
    counts = np.bincount(y)
    w = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
    return w / w.sum() * n_classes