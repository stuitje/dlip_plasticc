"""
evaluate_test.py — Evaluate ensemble of fold checkpoints on the test set.
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

from config import DEVICE, LOG_DIR, CHECKPOINT_DIR, FEATURES_DIR
from utils import HybridPlasticcNet, build_dataset
from dataset import obs_to_sequence

# ── Config ────────────────────────────────────────────────────────────────────
TEST_FEATURES_PATH = os.path.join(FEATURES_DIR, "features_v2_plasticc_test.h5")
TEST_NUM_CHUNKS    = 500   # matches sequential featurization
TEST_CHUNKS        = list(range(5))  # chunks 0-4
N_FOLDS            = 4

# ── Flat-weighted metric ──────────────────────────────────────────────────────
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

# ── Load augmented training data — scaler/le match the fold checkpoints ───────
print("Loading augmented training data...")
X_aug, y_aug, le, scaler, aug_ids, aug_cols = build_dataset()
n_classes   = len(le.classes_)
class_names = [str(c) for c in le.classes_]
feature_dim = X_aug.shape[1]
print(f"Feature dim: {feature_dim}  Classes: {n_classes}")

# ── Load and preprocess test features ─────────────────────────────────────────
print("\nLoading test features...")
test_raw = pd.read_hdf(TEST_FEATURES_PATH, key="raw_features")
print(f"  Test objects: {len(test_raw)}")

# Add missingness indicators and align to augmented column order
aug_raw  = pd.read_hdf(os.path.join(FEATURES_DIR,
                        "features_v2_plasticc_augmented.h5"), key="raw_features")
nan_cols = aug_raw.columns[aug_raw.isnull().any()]
indicators = pd.DataFrame(
    {f"{col}_missing": test_raw[col].isnull().astype(float)
     for col in nan_cols if col in test_raw.columns},
    index=test_raw.index
)
test_df = pd.concat([test_raw, indicators], axis=1)
test_df = test_df.reindex(columns=aug_cols, fill_value=0.0).fillna(0.0)
X_test  = scaler.transform(test_df.values).astype(np.float32)
X_test  = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  Test features shape: {X_test.shape}")

# ── Load test labels ──────────────────────────────────────────────────────────
print("Loading test labels...")
meta_list = []
for chunk in TEST_CHUNKS:
    try:
        m = avocado.load("plasticc_test", chunk=chunk,
                          num_chunks=TEST_NUM_CHUNKS,
                          metadata_only=True).metadata
        meta_list.append(m)
    except Exception as e:
        print(f"  Chunk {chunk} failed: {e}")
test_meta = pd.concat(meta_list)
test_meta = test_meta.loc[test_meta.index.isin(test_df.index)]
known     = test_meta["class"].isin(set(le.classes_))
test_meta = test_meta[known]

idx_map  = {oid: i for i, oid in enumerate(test_df.index)}
test_idx = [idx_map[oid] for oid in test_meta.index if oid in idx_map]
X_test_k = X_test[test_idx]
y_test_k = le.transform(test_meta["class"].values)
print(f"  Known class objects: {len(y_test_k)}")

# ── Load test observations ────────────────────────────────────────────────────
print("Loading test observations...")
test_obs = {}
for chunk in TEST_CHUNKS:
    try:
        d = avocado.load("plasticc_test", chunk=chunk,
                          num_chunks=TEST_NUM_CHUNKS)
        for obj in d.objects:
            oid = obj.metadata["object_id"]
            if oid in test_meta.index:
                test_obs[oid] = obj.observations
    except Exception as e:
        print(f"  Chunk {chunk} failed: {e}")
print(f"  Loaded observations for {len(test_obs):,} objects")

seqs, masks = [], []
for oid in test_meta.index:
    if oid in test_obs:
        seq, mask = obs_to_sequence(test_obs[oid])
    else:
        seq  = np.zeros((350, 5), dtype=np.float32)
        mask = np.zeros(350,      dtype=np.float32)
    seqs.append(seq)
    masks.append(mask)

seqs  = np.array(seqs,  dtype=np.float32)
masks = np.array(masks, dtype=np.float32)

ds     = TensorDataset(torch.tensor(seqs), torch.tensor(masks),
                       torch.tensor(X_test_k),
                       torch.tensor(y_test_k, dtype=torch.long))
loader = DataLoader(ds, batch_size=64, num_workers=0)

# ── Ensemble fold checkpoints ─────────────────────────────────────────────────
print(f"\nEnsembling {N_FOLDS} fold checkpoints...")
all_fold_probs = []

for fold in range(1, N_FOLDS + 1):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"fold_{fold}.pt")
    if not os.path.exists(ckpt_path):
        print(f"  Fold {fold}: not found, skipping")
        continue
    model = HybridPlasticcNet(feature_dim, n_classes).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    fold_probs = []
    with torch.no_grad():
        for seq, mask, feats, labels in loader:
            logits = model(seq.to(DEVICE), mask.to(DEVICE), feats.to(DEVICE))
            fold_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    all_fold_probs.append(np.vstack(fold_probs))
    print(f"  Fold {fold}: loaded ✓")

avg_probs  = np.mean(all_fold_probs, axis=0)
all_preds  = avg_probs.argmax(axis=1)
all_labels = y_test_k

# ── Metrics ───────────────────────────────────────────────────────────────────
flat_ll  = flat_weighted_logloss(all_labels, avg_probs, n_classes)
accuracy = (all_preds == all_labels).mean()

print("\n" + "=" * 65)
print(f"TEST SET RESULTS ({N_FOLDS}-fold ensemble)")
print("=" * 65)
print(f"Flat-weighted log-loss (Boone eq2) : {flat_ll:.4f}  (Boone: 0.468)")
print(f"Accuracy                            : {accuracy:.4f}")
print("=" * 65)

present_labels = sorted(np.unique(all_labels))
present_names  = [class_names[i] for i in present_labels]
print("\nClassification report:")
print(classification_report(all_labels, all_preds, labels=present_labels,
                            target_names=present_names, zero_division=0))

print("Per-class log-loss:")
for i, cls in enumerate(class_names):
    mask = all_labels == i
    if mask.sum() == 0:
        continue
    p   = np.clip(avg_probs[mask, i], 1e-15, 1.0)
    ll  = -np.mean(np.log(p))
    acc = (all_preds[mask] == i).mean()
    print(f"  Class {cls:>3s}: logloss={ll:.4f}  acc={acc:.3f}  (n={mask.sum()})")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm      = confusion_matrix(all_labels, all_preds, labels=present_labels)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=present_names, yticklabels=present_names, ax=axes[0])
axes[0].set_title(f"Test - {N_FOLDS}-fold ensemble (counts)")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Oranges",
            xticklabels=present_names, yticklabels=present_names, ax=axes[1])
axes[1].set_title(f"Test - {N_FOLDS}-fold ensemble (normalized)")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
plt.tight_layout()
os.makedirs(LOG_DIR, exist_ok=True)
out = os.path.join(LOG_DIR, "confusion_matrix_test_ensemble.png")
plt.savefig(out, dpi=150)
print(f"\nConfusion matrix saved to {out}")