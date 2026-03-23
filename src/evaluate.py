import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix, classification_report, log_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from utils import PlasticcNet, build_dataset, DEVICE

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/plasticc_nn_v3.pt"

# ── Data ──────────────────────────────────────────────────────────────────────
X, y, le, scaler = build_dataset()
n_classes   = len(le.classes_)
class_names = [str(c) for c in le.classes_]

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_t, y_t)
n_val   = int(0.15 * len(dataset))
n_train = len(dataset) - n_val
_, val_ds = random_split(dataset, [n_train, n_val],
                         generator=torch.Generator().manual_seed(42))
val_loader = DataLoader(val_ds, batch_size=256)

# ── Load model ────────────────────────────────────────────────────────────────
model = PlasticcNet(X.shape[1], n_classes).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ── Evaluate ──────────────────────────────────────────────────────────────────
all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for xb, yb in val_loader:
        logits = model(xb.to(DEVICE))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(yb.numpy())

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
plt.savefig("logs/confusion_matrix.png", dpi=150)
print("\nConfusion matrix saved to logs/confusion_matrix.png")

# ── Per-class accuracy ────────────────────────────────────────────────────────
print("\nPer-class accuracy:")
for i, cls in enumerate(class_names):
    mask = all_labels == i
    if mask.sum() == 0:
        continue
    acc = (all_preds[mask] == all_labels[mask]).mean()
    print(f"  Class {cls:>3s}: {acc:.3f}  (n={mask.sum()})")