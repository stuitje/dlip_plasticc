import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
import avocado
import math

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_PATH    = "/scratch/s4339150/plasticc/features/features_v1_plasticc_train.h5"
NUM_CHUNKS       = 8
NUM_TRAIN_CHUNKS = 5
DROPOUT          = 0.4
SEQ_LEN          = 350
D_MODEL          = 64
N_HEADS          = 4
N_LAYERS         = 2
FFN_DIM          = 128
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

# ── Time-based Positional Encoding ───────────────────────────────────────────
class TimePositionalEncoding(nn.Module):
    """
    Encodes the actual observation time (not sequence index) using sinusoidal
    functions. This captures irregular cadence — a 50-day gap is treated
    differently from a 1-day gap, unlike standard index-based PE.

    Input:  times [B, T]  (normalized to [0, 1])
    Output: PE    [B, T, d_model]
    """
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.d_model = d_model
        # Frequency terms: shape [d_model//2]
        i = torch.arange(0, d_model // 2, dtype=torch.float32)
        self.register_buffer("div_term", torch.exp(i * (-math.log(10000.0) / d_model)))

    def forward(self, times):
        # times: [B, T], values in [0, 1]
        # Scale to [0, 10000] so sinusoids span meaningful range
        t = times.unsqueeze(-1) * 10000.0          # [B, T, 1]
        sin_enc = torch.sin(t * self.div_term)      # [B, T, d_model//2]
        cos_enc = torch.cos(t * self.div_term)      # [B, T, d_model//2]
        pe = torch.cat([sin_enc, cos_enc], dim=-1)  # [B, T, d_model]
        return pe

# ── Transformer branch ────────────────────────────────────────────────────────
class LCTransformer(nn.Module):
    """
    Transformer encoder for raw light curve sequences with time-based PE.
    Input:  seq  [B, T, 5]  (time_norm, flux_norm, flux_err_norm, detected, band_id)
            mask [B, T]     (1=real, 0=pad)
    Output: [B, D_MODEL]    (mean-pooled)
    """
    def __init__(self, in_features=5, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        # Project input features to d_model (excluding time which goes to PE)
        # We use all 5 features in the projection, then add PE on top
        self.input_proj = nn.Linear(in_features, d_model)
        self.time_pe    = TimePositionalEncoding(d_model)
        self.dropout    = nn.Dropout(dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, mask):
        # x:    [B, T, 5]  — first feature is normalized time
        # mask: [B, T]
        times = x[:, :, 0]                          # [B, T] normalized time
        x_proj = self.input_proj(x)                 # [B, T, d_model]
        pe     = self.time_pe(times)                # [B, T, d_model]
        x_proj = self.dropout(x_proj + pe)          # [B, T, d_model]
        pad_mask = (mask == 0)                       # True = ignore (padding)
        x_proj = self.encoder(x_proj, src_key_padding_mask=pad_mask)
        # Mean pool over real timesteps only
        mask_expanded = mask.unsqueeze(-1)           # [B, T, 1]
        out = (x_proj * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return out                                   # [B, d_model]

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
        combined = torch.cat([lc_emb, feat_emb], dim=1)
        return self.head(combined)

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
        d = avocado.load("plasticc_train", chunk=chunk,
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
    object_ids = df.index.tolist()
    return X, y, le, scaler, object_ids

def class_weights(y, n_classes, device):
    counts = np.bincount(y)
    w = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
    return w / w.sum() * n_classes