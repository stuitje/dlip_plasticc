import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
import os

from config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, EARLY_STOP_PATIENCE,
    FOCAL_GAMMA, CHECKPOINT_HYBRID, DEVICE
)
from utils import HybridPlasticcNet, FocalLoss, build_dataset, class_weights
from dataset import PlasticcDataset, load_observations

print(f"Using device: {DEVICE}")

# ── Data ──────────────────────────────────────────────────────────────────────
X, y, le, scaler, object_ids = build_dataset()
n_classes = len(le.classes_)
print(f"Classes ({n_classes}): {le.classes_}")

obs_dict   = load_observations()
valid_ids  = [oid for oid in object_ids if oid in obs_dict]
valid_idx  = [object_ids.index(oid) for oid in valid_ids]
X_valid    = X[valid_idx]
y_valid    = y[valid_idx]
print(f"Objects with observations: {len(valid_ids)}")

# ── Train/val split ───────────────────────────────────────────────────────────
n_total   = len(valid_ids)
n_val     = int(0.15 * n_total)
n_train   = n_total - n_val
rng       = np.random.default_rng(42)
idx_perm  = rng.permutation(n_total)
train_idx = idx_perm[:n_train]
val_idx   = idx_perm[n_train:]

train_ds = PlasticcDataset(
    [valid_ids[i] for i in train_idx], obs_dict,
    X_valid[train_idx], y_valid[train_idx], augment=True
)
val_ds = PlasticcDataset(
    [valid_ids[i] for i in val_idx], obs_dict,
    X_valid[val_idx], y_valid[val_idx], augment=False
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train: {n_train}  Val: {n_val}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = HybridPlasticcNet(X.shape[1], n_classes).to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

cw        = class_weights(y_valid, n_classes, DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=cw)

# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train(train)
    total_loss, correct, n = 0, 0, 0
    all_probs, all_labels  = [], []
    with torch.set_grad_enabled(train):
        for seq, mask, feats, labels in loader:
            seq, mask     = seq.to(DEVICE),   mask.to(DEVICE)
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            logits = model(seq, mask, feats)
            loss   = criterion(logits, labels)
            if train:
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
    logloss    = log_loss(all_labels, all_probs, labels=list(range(n_classes)))
    return total_loss / n, correct / n, logloss

os.makedirs(os.path.dirname(CHECKPOINT_HYBRID), exist_ok=True)
os.makedirs("logs", exist_ok=True)

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
        torch.save(model.state_dict(), CHECKPOINT_HYBRID)
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
print(f"Checkpoint saved to {CHECKPOINT_HYBRID}")