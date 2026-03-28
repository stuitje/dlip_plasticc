"""
evaluate_test.py — Evaluate best ensemble model on a chunk of the test set.
Uses the true labels from plasticc_test.h5 metadata to compute metrics.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import avocado
import os

from config import DEVICE, LOG_DIR, CHECKPOINT_DIR, SCRATCH_DIR
from utils import HybridPlasticcNet, build_dataset

# ── Config ────────────────────────────────────────────────────────────────────
TEST_FEATURES_PATH  = "/scratch/s4339150/plasticc/features/features_v2_plasticc_test.h5"
TRAIN_FEATURES_PATH = "/scratch/s4339150/plasticc/features/features_v2_plasticc_train.h5"
CHECKPOINT          = os.path.join(CHECKPOINT_DIR, "plasticc_hybrid_augmented.pt")
NUM_CHUNKS          = 3500
TEST_CHUNK          = 0

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

# ── Load training data to get scaler, le, and column structure ────────────────
print("Loading training data for preprocessing params...")
X_train, y_train, le, scaler, train_object_ids = build_dataset()
n_classes   = len(le.classes_)
class_names = [str(c) for c in le.classes_]
print(f"  Training features: {X_train.shape}")

# Get training column order after preprocessing
train_raw = pd.read_hdf(TRAIN_FEATURES_PATH, key="raw_features")
nan_cols_train = train_raw.columns[train_raw.isnull().any()]
indicator_cols = [f"{col}_missing" for col in nan_cols_train]
train_col_order = list(train_raw.columns) + indicator_cols
print(f"  Training column order: {len(train_col_order)} columns")

# ── Load and preprocess test features ─────────────────────────────────────────
print("\nLoading test features...")
test_raw = pd.read_hdf(TEST_FEATURES_PATH, key="raw_features")
print(f"  Test raw shape: {test_raw.shape}")

# Add missingness indicators using TRAINING nan columns
indicators = pd.DataFrame(
    {f"{col}_missing": test_raw[col].isnull().astype(float)
     for col in nan_cols_train if col in test_raw.columns},
    index=test_raw.index
)
test_df = pd.concat([test_raw, indicators], axis=1)

# Reindex to match training column order exactly
test_df = test_df.reindex(columns=train_col_order, fill_value=0.0)
test_df = test_df.fillna(test_df.median())
print(f"  Test preprocessed shape: {test_df.shape}")

# Apply training scaler
X_test = scaler.transform(test_df.values).astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# ── Load test labels ──────────────────────────────────────────────────────────
print("\nLoading test labels...")
test_meta = avocado.load("plasticc_test", chunk=TEST_CHUNK,
                          num_chunks=NUM_CHUNKS, metadata_only=True).metadata
test_meta = test_meta.loc[test_df.index]
print(f"  Class distribution:\n{test_meta['class'].value_counts()}")

# Filter to known classes only
known_classes = set(le.classes_)
mask_known    = test_meta["class"].isin(known_classes)
print(f"\n  Objects with known classes: {mask_known.sum()} / {len(mask_known)}")

X_test_known = X_test[mask_known]
y_test_known = le.transform(test_meta["class"][mask_known].values)

# ── Load raw observations for transformer branch ──────────────────────────────
print("\nLoading test observations...")
test_avocado = avocado.load("plasticc_test", chunk=TEST_CHUNK,
                             num_chunks=NUM_CHUNKS)
obs_dict = {obj.metadata["object_id"]: obj.observations
            for obj in test_avocado.objects}
print(f"  Loaded {len(obs_dict)} objects")

from dataset import obs_to_sequence

test_oids = list(test_df.index[mask_known])
sequences, masks_arr = [], []
for oid in test_oids:
    if oid in obs_dict:
        seq, mask = obs_to_sequence(obs_dict[oid])
    else:
        seq  = np.zeros((350, 5), dtype=np.float32)
        mask = np.zeros(350,      dtype=np.float32)
    sequences.append(seq)
    masks_arr.append(mask)

sequences = np.array(sequences, dtype=np.float32)
masks_arr = np.array(masks_arr, dtype=np.float32)

# ── Load model ────────────────────────────────────────────────────────────────
model = HybridPlasticcNet(X_train.shape[1], n_classes).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()
print(f"\nLoaded: {CHECKPOINT}")

ds     = TensorDataset(torch.tensor(sequences),
                       torch.tensor(masks_arr),
                       torch.tensor(X_test_known),
                       torch.tensor(y_test_known, dtype=torch.long))
loader = DataLoader(ds, batch_size=64, num_workers=0)

all_probs, all_preds, all_labels = [], [], []
with torch.no_grad():
    for seq, mask, feats, labels in loader:
        logits = model(seq.to(DEVICE), mask.to(DEVICE), feats.to(DEVICE))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(labels.numpy())

all_probs  = np.vstack(all_probs)
all_preds  = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# ── Metrics ───────────────────────────────────────────────────────────────────
flat_ll  = flat_weighted_logloss(all_labels, all_probs, n_classes)
accuracy = (all_preds == all_labels).mean()

print("\n" + "=" * 60)
print("TEST SET RESULTS (chunk 0 of 3500)")
print("=" * 60)
print(f"Flat-weighted log-loss (Boone eq2) : {flat_ll:.4f}  (Boone: 0.468)")
print(f"Accuracy                            : {accuracy:.4f}")
print("=" * 60)

# Use only labels present in this chunk
present_labels = sorted(np.unique(all_labels))
present_names  = [class_names[i] for i in present_labels]
print("\nClassification report:")
print(classification_report(all_labels, all_preds,
                            labels=present_labels,
                            target_names=present_names,
                            zero_division=0))

print("Per-class log-loss:")
for i, cls in enumerate(class_names):
    mask = all_labels == i
    if mask.sum() == 0:
        continue
    p   = np.clip(all_probs[mask, i], 1e-15, 1.0)
    ll  = -np.mean(np.log(p))
    acc = (all_preds[mask] == i).mean()
    print(f"  Class {cls:>3s}: logloss={ll:.4f}  acc={acc:.3f}  (n={mask.sum()})")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm      = confusion_matrix(all_labels, all_preds, labels=present_labels)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=present_names, yticklabels=present_names, ax=axes[0])
axes[0].set_title("Test Confusion Matrix (counts)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Oranges",
            xticklabels=present_names, yticklabels=present_names, ax=axes[1])
axes[1].set_title("Test Confusion Matrix (normalized)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")
plt.tight_layout()
os.makedirs(LOG_DIR, exist_ok=True)
out = os.path.join(LOG_DIR, "confusion_matrix_test.png")
plt.savefig(out, dpi=150)
print(f"\nConfusion matrix saved to {out}")