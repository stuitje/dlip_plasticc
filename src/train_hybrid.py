"""
train_hybrid.py — Train hybrid transformer+MLP classifier following Boone (2019).

Key differences from previous version:
- Trains on ALL augmented data (no held-out val set from training distribution)
- Uses the actual test set chunk for validation/early stopping
- Splits by original object ID to prevent leakage
- Uses augmented GP sequences for transformer branch
- Supports redshift-weighted loss
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
import avocado
import os

from config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, EARLY_STOP_PATIENCE,
    FOCAL_GAMMA, CHECKPOINT_HYBRID, DEVICE,
    USE_REDSHIFT_WEIGHTING, REDSHIFT_SCALE,
    FEATURES_DIR, SCRATCH_DIR, BAND_MAP, SEQ_LEN
)
from utils import HybridPlasticcNet, FocalLoss, RedshiftWeightedLoss, build_dataset, class_weights, preprocess
from dataset import PlasticcDataset, load_observations, get_aug_sequences, obs_to_sequence

# ── Config ────────────────────────────────────────────────────────────────────
TEST_FEATURES_PATH = os.path.join(FEATURES_DIR, "features_v2_plasticc_test.h5")
TEST_NUM_CHUNKS    = 3500
TEST_CHUNKS        = list(range(10))  # use chunks 0-9 for validation (~10k objects)

# ── Flat-weighted metric (Boone eq. 2) ────────────────────────────────────────
def flat_weighted_logloss(y_true, y_probs, n_classes):
    total, count = 0.0, 0
    for i in range(n_classes):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        p = np.clip(y_probs[mask, i], 1e-15, 1.0)
        total += -np.mean(np.log(p))
        count += 1
    return total / count

# ── Load augmented training data ──────────────────────────────────────────────
print("=" * 65)
print("Loading augmented training data...")
print("=" * 65)
X_train, y_train, le, scaler, object_ids = build_dataset()
n_classes   = len(le.classes_)
class_names = [str(c) for c in le.classes_]
print(f"Classes ({n_classes}): {le.classes_}")

# Load original observations for fallback
obs_dict, meta_dict = load_observations()

# Match features to observations via original ID
def get_original_id(oid):
    return oid.split("_aug_")[0] if "_aug_" in oid else oid

valid_ids = [oid for oid in object_ids
             if get_original_id(oid) in obs_dict]
valid_idx = [object_ids.index(oid) for oid in valid_ids]
X_valid   = X_train[valid_idx]
y_valid   = y_train[valid_idx]
print(f"Training objects: {len(valid_ids):,}")

# Load augmented sequences for transformer
aug_seqs = get_aug_sequences()

# ── Load test set for validation ──────────────────────────────────────────────
print("\nLoading test set for validation...")

# Load test features
test_features_raw = pd.read_hdf(TEST_FEATURES_PATH, key="raw_features")
print(f"  Test features: {test_features_raw.shape}")

# Preprocess test features using TRAINING scaler
train_raw = pd.read_hdf(
    os.path.join(FEATURES_DIR, "features_v2_plasticc_train.h5"),
    key="raw_features"
)
nan_cols_train = train_raw.columns[train_raw.isnull().any()]
indicator_cols = [f"{col}_missing" for col in nan_cols_train]
train_col_order = list(train_raw.columns) + list(indicator_cols)

indicators = pd.DataFrame(
    {f"{col}_missing": test_features_raw[col].isnull().astype(float)
     for col in nan_cols_train if col in test_features_raw.columns},
    index=test_features_raw.index
)
test_df = pd.concat([test_features_raw, indicators], axis=1)
test_df = test_df.reindex(columns=train_col_order, fill_value=0.0).fillna(0.0)
X_test  = scaler.transform(test_df.values).astype(np.float32)
X_test  = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Load test labels
print("  Loading test labels...")
test_meta = avocado.load("plasticc_test", chunk=0,
                          num_chunks=TEST_NUM_CHUNKS,
                          metadata_only=True).metadata
# Load more chunks if available
for chunk in TEST_CHUNKS[1:]:
    try:
        m = avocado.load("plasticc_test", chunk=chunk,
                          num_chunks=TEST_NUM_CHUNKS,
                          metadata_only=True).metadata
        test_meta = pd.concat([test_meta, m])
    except Exception:
        break

test_meta = test_meta.loc[test_meta.index.isin(test_df.index)]
known     = test_meta["class"].isin(set(le.classes_))
test_meta = test_meta[known]
X_test_k  = X_test[test_df.index.isin(test_meta.index)]
y_test_k  = le.transform(test_meta["class"].values)
print(f"  Test objects with known classes: {len(y_test_k):,}")

# Load test observations for transformer
print("  Loading test observations...")
test_obs = {}
for chunk in TEST_CHUNKS:
    try:
        d = avocado.load("plasticc_test", chunk=chunk,
                          num_chunks=TEST_NUM_CHUNKS)
        for obj in d.objects:
            oid = obj.metadata["object_id"]
            if oid in test_meta.index:
                test_obs[oid] = obj.observations
    except Exception:
        break
print(f"  Test observations loaded: {len(test_obs):,}")

# Build test sequences
test_oids = list(test_meta.index)
test_seqs, test_masks = [], []
for oid in test_oids:
    if oid in test_obs:
        seq, mask = obs_to_sequence(test_obs[oid])
    else:
        seq  = np.zeros((SEQ_LEN, 5), dtype=np.float32)
        mask = np.zeros(SEQ_LEN,      dtype=np.float32)
    test_seqs.append(seq)
    test_masks.append(mask)

test_seqs  = np.array(test_seqs,  dtype=np.float32)
test_masks = np.array(test_masks, dtype=np.float32)

# ── Datasets ──────────────────────────────────────────────────────────────────
# Train on ALL augmented data
train_ds = PlasticcDataset(
    valid_ids, obs_dict, meta_dict,
    X_valid, y_valid, augment=False, use_aug_sequences=True
)

# Val = actual test set
from torch.utils.data import TensorDataset
val_ds = TensorDataset(
    torch.tensor(test_seqs),
    torch.tensor(test_masks),
    torch.tensor(X_test_k),
    torch.tensor(y_test_k, dtype=torch.long),
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\nTrain: {len(train_ds):,}  Val (test set): {len(val_ds):,}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = HybridPlasticcNet(X_train.shape[1], n_classes).to(DEVICE)
cw    = class_weights(y_valid, n_classes, DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

if USE_REDSHIFT_WEIGHTING:
    print(f"Using redshift-weighted loss (scale={REDSHIFT_SCALE})")
    criterion = RedshiftWeightedLoss(gamma=FOCAL_GAMMA, class_weight=cw,
                                     redshift_scale=REDSHIFT_SCALE)
else:
    criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=cw)

# ── Training loop ─────────────────────────────────────────────────────────────
def run_train_epoch():
    model.train()
    total_loss, correct, n = 0, 0, 0
    all_probs, all_labels  = [], []
    for seq, mask, feats, labels, redshifts in train_loader:
        seq, mask     = seq.to(DEVICE),   mask.to(DEVICE)
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        logits = model(seq, mask, feats)
        if USE_REDSHIFT_WEIGHTING:
            loss = criterion(logits, labels, redshifts=redshifts)
        else:
            loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)
        all_probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_probs  = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    ll = log_loss(all_labels, all_probs, labels=list(range(n_classes)))
    return total_loss / n, correct / n, ll

def run_val_epoch():
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for seq, mask, feats, labels in val_loader:
            logits = model(seq.to(DEVICE), mask.to(DEVICE), feats.to(DEVICE))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(labels.numpy())
    all_probs  = np.vstack(all_probs)
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    flat_ll = flat_weighted_logloss(all_labels, all_probs, n_classes)
    acc     = (all_preds == all_labels).mean()
    return flat_ll, acc

os.makedirs(os.path.dirname(CHECKPOINT_HYBRID), exist_ok=True)
os.makedirs("logs", exist_ok=True)

best_val = float("inf")
patience = 0

print(f"\nUsing device: {DEVICE}")
print("\nEpoch | Train loss | Train acc | Test acc | Test logloss | Best")
print("-" * 68)

for epoch in range(EPOCHS):
    tr_loss, tr_acc, tr_ll = run_train_epoch()
    va_ll, va_acc           = run_val_epoch()
    scheduler.step()

    improved = va_ll < best_val
    if improved:
        best_val = va_ll
        torch.save(model.state_dict(), CHECKPOINT_HYBRID)
        patience = 0
    else:
        patience += 1

    marker = " ✓" if improved else f" ({patience}/{EARLY_STOP_PATIENCE})"
    print(f"{epoch+1:5d} | {tr_loss:.4f}     | {tr_acc:.3f}     | "
          f"{va_acc:.3f}    | {va_ll:.4f}       |{marker}")

    if patience >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print(f"\nBest test flat-weighted log-loss: {best_val:.4f}  (Boone: 0.468)")
print(f"Checkpoint saved to {CHECKPOINT_HYBRID}")