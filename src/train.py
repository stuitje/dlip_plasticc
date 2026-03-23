import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import log_loss
import os
from utils import PlasticcNet, FocalLoss, build_dataset, class_weights, DEVICE

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE          = 128
EPOCHS              = 200
LR                  = 1e-3
WEIGHT_DECAY        = 1e-4
EARLY_STOP_PATIENCE = 15
FOCAL_GAMMA         = 2.0
CHECKPOINT_PATH     = "checkpoints/plasticc_nn_v3.pt"

print(f"Using device: {DEVICE}")

# ── Data ──────────────────────────────────────────────────────────────────────
X, y, le, scaler = build_dataset()
n_classes = len(le.classes_)
print(f"Classes ({n_classes}): {le.classes_}")

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_t, y_t)
n_val   = int(0.15 * len(dataset))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256)
print(f"Train: {n_train}  Val: {n_val}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = PlasticcNet(X.shape[1], n_classes).to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

cw        = class_weights(y, n_classes, DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=cw)

# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train(train)
    total_loss, correct, n = 0, 0, 0
    all_probs, all_labels  = [], []
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss   = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            n          += len(yb)
            all_probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    all_probs  = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    logloss    = log_loss(all_labels, all_probs, labels=list(range(n_classes)))
    return total_loss / n, correct / n, logloss

os.makedirs("checkpoints", exist_ok=True)
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
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 5 == 0:
        marker = " ✓" if improved else f" ({patience_counter}/{EARLY_STOP_PATIENCE})"
        print(f"{epoch+1:5d} | {tr_loss:.4f}     | {tr_acc:.3f}     | "
              f"{va_acc:.3f}   | {va_ll:.4f}      |{marker}")

    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print(f"\nBest val log-loss: {best_val_logloss:.4f}")
print(f"Checkpoint saved to {CHECKPOINT_PATH}")