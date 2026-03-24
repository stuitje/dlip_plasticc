import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, log_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import CHECKPOINT_HYBRID, DEVICE, LOG_DIR
from utils import HybridPlasticcNet, build_dataset
from dataset import PlasticcDataset, load_observations

# ── Data ──────────────────────────────────────────────────────────────────────
X, y, le, scaler, object_ids = build_dataset()
n_classes   = len(le.classes_)
class_names = [str(c) for c in le.classes_]

obs_dict  = load_observations()
valid_ids = [oid for oid in object_ids if oid in obs_dict]
valid_idx = [object_ids.index(oid) for oid in valid_ids]
X_valid   = X[valid_idx]
y_valid   = y[valid_idx]

# ── Same split as train_hybrid.py ─────────────────────────────────────────────
n_total = len(valid_ids)
n_val   = int(0.15 * n_total)
n_train = n_total - n_val
rng     = np.random.default_rng(42)
idx_perm = rng.permutation(n_total)
val_idx_split = idx_perm[n_train:]

val_ds = PlasticcDataset(
    [valid_ids[i] for i in val_idx_split], obs_dict,
    X_valid[val_idx_split], y_valid[val_idx_split], augment=False
)
val_loader = DataLoader(val_ds, batch_size=64, num_workers=0)

# ── Load model ────────────────────────────────────────────────────────────────
model = HybridPlasticcNet(X.shape[1], n_classes).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_HYBRID, map_location=DEVICE))
model.eval()
print(f"Loaded: {CHECKPOINT_HYBRID}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for seq, mask, feats, labels in val_loader:
        seq, mask     = seq.to(DEVICE),   mask.to(DEVICE)
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        logits = model(seq, mask, feats)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

all_probs  = np.vstack(all_probs)
all_preds  = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

print("=" * 60)
print(f"Val log-loss : {log_loss(all_labels, all_probs):.4f}")
print(f"Val accuracy : {(all_preds == all_labels).mean():.4f}")
print("=" * 60)
print("\nClassification report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ── Confusion matrix ──────────────────────────────────────────────────────────
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
os.makedirs(LOG_DIR, exist_ok=True)
plt.savefig(os.path.join(LOG_DIR, "confusion_matrix_hybrid.png"), dpi=150)
print(f"\nConfusion matrix saved to {LOG_DIR}/confusion_matrix_hybrid.png")

# ── Per-class accuracy ────────────────────────────────────────────────────────
print("\nPer-class accuracy:")
for i, cls in enumerate(class_names):
    mask = all_labels == i
    if mask.sum() == 0:
        continue
    acc = (all_preds[mask] == all_labels[mask]).mean()
    print(f"  Class {cls:>3s}: {acc:.3f}  (n={mask.sum()})")