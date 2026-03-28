"""
transformer.py — Standalone transformer classifier on raw light curves.

Trains directly on the augmented dataset (plasticc_augmented) using only
raw light curve sequences — no avocado features needed.

Splits by original object ID to prevent data leakage — all augmented
versions of an object stay in the same train/val split.

Usage:
    python src/transformer.py              # train
    python src/transformer.py --evaluate   # evaluate saved checkpoint
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import math
import avocado

from config import (
    BAND_MAP, SEQ_LEN, SCRATCH_DIR, NUM_CHUNKS,
    D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, DROPOUT,
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
    EARLY_STOP_PATIENCE, FOCAL_GAMMA, DEVICE, LOG_DIR, CHECKPOINT_DIR
)

CHECKPOINT_TRANSFORMER = os.path.join(CHECKPOINT_DIR, "transformer_only.pt")
AUG_DATASET_NAME       = "plasticc_augmented"
AUG_NUM_CHUNKS         = 48

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_original_id(oid):
    """Strip augmentation suffix to get original object ID."""
    return oid.split("_aug_")[0] if "_aug_" in oid else oid

# ── Sequence conversion ───────────────────────────────────────────────────────
def obs_to_sequence(obs):
    obs    = obs.sort_values("time")
    t      = obs["time"].values.astype(np.float32)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)
    flux     = obs["flux"].values.astype(np.float32)
    flux_err = obs["flux_error"].values.astype(np.float32)
    flux_std = flux.std() + 1e-8
    detected = obs["detected"].values.astype(np.float32)
    band_ids = obs["band"].map(BAND_MAP).fillna(0).values.astype(np.float32)
    seq = np.stack([t_norm, flux / flux_std, flux_err / flux_std,
                    detected, band_ids], axis=1)
    T = len(seq)
    if T >= SEQ_LEN:
        return seq[:SEQ_LEN], np.ones(SEQ_LEN, dtype=np.float32)
    pad  = np.zeros((SEQ_LEN - T, 5), dtype=np.float32)
    mask = np.array([1.0] * T + [0.0] * (SEQ_LEN - T), dtype=np.float32)
    return np.vstack([seq, pad]), mask

# ── Dataset ───────────────────────────────────────────────────────────────────
class LCOnlyDataset(Dataset):
    def __init__(self, sequences, masks, labels):
        self.sequences = sequences
        self.masks     = masks
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.masks[idx],     dtype=torch.float32),
            torch.tensor(self.labels[idx],    dtype=torch.long),
        )

# ── Time-based Positional Encoding ───────────────────────────────────────────
class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        i = torch.arange(0, d_model // 2, dtype=torch.float32)
        self.register_buffer("div_term",
                             torch.exp(i * (-math.log(10000.0) / d_model)))

    def forward(self, times):
        t = times.unsqueeze(-1) * 10000.0
        return torch.cat([torch.sin(t * self.div_term),
                          torch.cos(t * self.div_term)], dim=-1)

# ── Transformer classifier ────────────────────────────────────────────────────
class LCTransformerClassifier(nn.Module):
    def __init__(self, n_classes, in_features=5, d_model=D_MODEL,
                 n_heads=N_HEADS, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
                 dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.time_pe    = TimePositionalEncoding(d_model)
        self.dropout    = nn.Dropout(dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x, mask):
        times    = x[:, :, 0]
        x_proj   = self.input_proj(x)
        pe       = self.time_pe(times)
        x_proj   = self.dropout(x_proj + pe)
        pad_mask = (mask == 0)
        x_proj   = self.encoder(x_proj, src_key_padding_mask=pad_mask)
        mask_exp = mask.unsqueeze(-1)
        out = (x_proj * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        return self.head(out)

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
        loss    = -(1 - p_t) ** self.gamma * log_p_t
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()

# ── Data loading ──────────────────────────────────────────────────────────────
def load_augmented_dataset():
    cache_path = os.path.join(SCRATCH_DIR, "augmented_sequences.pkl")

    if os.path.exists(cache_path):
        print(f"Loading sequence cache: {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        print(f"  Loaded {len(data['labels']):,} sequences")
        return data

    print("Converting augmented dataset to sequences (first time — will cache)...")
    sequences, masks, labels, object_ids = [], [], [], []

    for chunk in range(AUG_NUM_CHUNKS):
        print(f"  Chunk {chunk+1}/{AUG_NUM_CHUNKS}...", end="\r")
        try:
            d = avocado.load(AUG_DATASET_NAME, chunk=chunk,
                             num_chunks=AUG_NUM_CHUNKS)
            for obj in d.objects:
                try:
                    seq, mask = obs_to_sequence(obj.observations)
                    sequences.append(seq)
                    masks.append(mask)
                    labels.append(obj.metadata["class"])
                    object_ids.append(obj.metadata["object_id"])
                except Exception:
                    continue
        except Exception as e:
            print(f"\n  Chunk {chunk} failed: {e}")
            continue

    print(f"\nLoaded {len(sequences):,} sequences")

    le = LabelEncoder()
    y  = le.fit_transform(labels).astype(np.int64)

    data = {
        "sequences":  np.array(sequences,  dtype=np.float32),
        "masks":      np.array(masks,       dtype=np.float32),
        "labels":     y,
        "classes":    le.classes_,
        "object_ids": np.array(object_ids),
    }

    print(f"Saving sequence cache...")
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Cache saved ({os.path.getsize(cache_path)/1e6:.0f} MB)")
    return data


def make_split(data):
    """
    Split by original object ID to prevent data leakage.
    All augmented versions of a real object go to the same split.
    """
    object_ids   = data["object_ids"]
    original_ids = np.array([get_original_id(oid) for oid in object_ids])
    unique_orig  = np.unique(original_ids)

    rng = np.random.default_rng(42)
    rng.shuffle(unique_orig)
    n_val_orig     = int(0.15 * len(unique_orig))
    val_originals  = set(unique_orig[-n_val_orig:])

    train_idx = np.where([get_original_id(oid) not in val_originals
                          for oid in object_ids])[0]
    val_idx   = np.where([get_original_id(oid) in val_originals
                          for oid in object_ids])[0]

    n_train_orig = len(unique_orig) - n_val_orig
    print(f"Split by original object:")
    print(f"  Train: {len(train_idx):,} sequences ({n_train_orig} original objects)")
    print(f"  Val:   {len(val_idx):,} sequences ({n_val_orig} original objects)")

    return train_idx, val_idx

# ── Training ──────────────────────────────────────────────────────────────────
def train():
    print(f"Using device: {DEVICE}")

    data      = load_augmented_dataset()
    sequences = data["sequences"]
    masks     = data["masks"]
    labels    = data["labels"]
    classes   = data["classes"]
    n_classes = len(classes)

    print(f"\nTotal sequences: {len(sequences):,}")
    print(f"Classes ({n_classes}): {classes}")

    train_idx, val_idx = make_split(data)

    # Class weights based on training set only
    train_labels  = labels[train_idx]
    counts_arr    = np.bincount(train_labels, minlength=n_classes)
    class_weights = torch.tensor(1.0 / np.maximum(counts_arr, 1),
                                 dtype=torch.float32).to(DEVICE)
    class_weights = class_weights / class_weights.sum() * n_classes

    train_ds = LCOnlyDataset(sequences[train_idx], masks[train_idx],
                              labels[train_idx])
    val_ds   = LCOnlyDataset(sequences[val_idx],   masks[val_idx],
                              labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    model     = LCTransformerClassifier(n_classes).to(DEVICE)
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=EPOCHS)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)

    def run_epoch(loader, train=True):
        model.train(train)
        total_loss, correct, n = 0, 0, 0
        all_probs, all_labels  = [], []
        with torch.set_grad_enabled(train):
            for seq, mask, lbl in loader:
                seq, mask, lbl = seq.to(DEVICE), mask.to(DEVICE), lbl.to(DEVICE)
                logits = model(seq, mask)
                loss   = criterion(logits, lbl)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                total_loss += loss.item() * len(lbl)
                correct    += (logits.argmax(1) == lbl).sum().item()
                n          += len(lbl)
                all_probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                all_labels.append(lbl.cpu().numpy())
        all_probs  = np.vstack(all_probs)
        all_labels = np.concatenate(all_labels)
        logloss    = log_loss(all_labels, all_probs,
                              labels=list(range(n_classes)))
        return total_loss / n, correct / n, logloss

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    best_val_logloss = float("inf")
    patience_counter = 0

    print("\nEpoch | Train loss | Train acc | Val acc | Val logloss | Best")
    print("-" * 65)

    for epoch in range(EPOCHS):
        tr_loss, tr_acc, tr_ll = run_epoch(train_loader, train=True)
        va_loss, va_acc, va_ll = run_epoch(val_loader,   train=False)
        scheduler.step()

        improved = va_ll < best_val_logloss
        if improved:
            best_val_logloss = va_ll
            torch.save({
                "model_state": model.state_dict(),
                "classes":     classes,
                "n_classes":   n_classes,
            }, CHECKPOINT_TRANSFORMER)
            patience_counter = 0
        else:
            patience_counter += 1

        marker = " ✓" if improved else f" ({patience_counter}/{EARLY_STOP_PATIENCE})"
        print(f"{epoch+1:5d} | {tr_loss:.4f}     | {tr_acc:.3f}     | "
              f"{va_acc:.3f}   | {va_ll:.4f}      |{marker}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nBest val log-loss: {best_val_logloss:.4f}")
    print(f"Checkpoint saved to {CHECKPOINT_TRANSFORMER}")


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate():
    print(f"Loading checkpoint: {CHECKPOINT_TRANSFORMER}")
    ckpt      = torch.load(CHECKPOINT_TRANSFORMER, map_location=DEVICE)
    classes   = ckpt["classes"]
    n_classes = ckpt["n_classes"]
    class_names = [str(c) for c in classes]

    data    = load_augmented_dataset()
    _, val_idx = make_split(data)

    val_ds     = LCOnlyDataset(data["sequences"][val_idx],
                                data["masks"][val_idx],
                                data["labels"][val_idx])
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    model = LCTransformerClassifier(n_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for seq, mask, lbl in val_loader:
            logits = model(seq.to(DEVICE), mask.to(DEVICE))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(lbl.numpy())

    all_probs  = np.vstack(all_probs)
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    print("=" * 60)
    print(f"Val log-loss : {log_loss(all_labels, all_probs):.4f}")
    print(f"Val accuracy : {(all_preds == all_labels).mean():.4f}")
    print("=" * 60)
    print(classification_report(all_labels, all_preds,
                                 target_names=class_names))

    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    plt.tight_layout()
    out_path = os.path.join(LOG_DIR, "confusion_matrix_transformer.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nConfusion matrix saved to {out_path}")

    print("\nPer-class accuracy:")
    for i, cls in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() == 0:
            continue
        acc = (all_preds[mask] == all_labels[mask]).mean()
        print(f"  Class {cls:>3s}: {acc:.3f}  (n={mask.sum():,})")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate saved checkpoint instead of training")
    args = parser.parse_args()

    if args.evaluate:
        evaluate()
    else:
        train()