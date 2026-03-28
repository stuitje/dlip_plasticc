"""
train_hybrid.py — Train hybrid transformer+MLP classifier following Boone (2019).

Uses 5-fold cross-validation on the augmented training set, splitting by
original object ID to prevent leakage. Final model trained on all data.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import os

from config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, EARLY_STOP_PATIENCE,
    FOCAL_GAMMA, CHECKPOINT_HYBRID, DEVICE,
    USE_REDSHIFT_WEIGHTING, REDSHIFT_SCALE, CHECKPOINT_DIR
)
from utils import HybridPlasticcNet, FocalLoss, RedshiftWeightedLoss, build_dataset, class_weights
from dataset import PlasticcDataset, load_observations, get_aug_sequences, get_original_id

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

# ── Data ──────────────────────────────────────────────────────────────────────
print(f"Using device: {DEVICE}")
print(f"Redshift weighting: {USE_REDSHIFT_WEIGHTING}")

X, y, le, scaler, object_ids = build_dataset()
n_classes   = len(le.classes_)
class_names = [str(c) for c in le.classes_]
print(f"Classes ({n_classes}): {le.classes_}")

obs_dict, meta_dict = load_observations()
aug_seqs = get_aug_sequences()

# Match features to observations via original ID
valid_ids = [oid for oid in object_ids
             if get_original_id(oid) in obs_dict]
valid_idx = [object_ids.index(oid) for oid in valid_ids]
X_valid   = X[valid_idx]
y_valid   = y[valid_idx]
print(f"Total sequences: {len(valid_ids):,}")

# ── Split by original object ID ───────────────────────────────────────────────
original_ids = np.array([get_original_id(oid) for oid in valid_ids])
unique_orig  = np.unique(original_ids)
print(f"Unique original objects: {len(unique_orig):,}")

# ── 5-fold CV ─────────────────────────────────────────────────────────────────
N_FOLDS = 5
kf      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_scores   = []
oof_probs     = np.zeros((len(valid_ids), n_classes))  # out-of-fold predictions

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

def train_model(train_idx, val_idx, fold=None):
    """Train one model on train_idx, validate on val_idx."""
    train_ds = PlasticcDataset(
        [valid_ids[i] for i in train_idx], obs_dict, meta_dict,
        X_valid[train_idx], y_valid[train_idx],
        augment=False, use_aug_sequences=True
    )
    val_ds = PlasticcDataset(
        [valid_ids[i] for i in val_idx], obs_dict, meta_dict,
        X_valid[val_idx], y_valid[val_idx],
        augment=False, use_aug_sequences=True
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model     = HybridPlasticcNet(X.shape[1], n_classes).to(DEVICE)
    cw        = class_weights(y_valid[train_idx], n_classes, DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=EPOCHS)
    if USE_REDSHIFT_WEIGHTING:
        criterion = RedshiftWeightedLoss(gamma=FOCAL_GAMMA, class_weight=cw,
                                         redshift_scale=REDSHIFT_SCALE)
    else:
        criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=cw)

    def run_epoch(loader, train=True):
        model.train(train)
        total_loss, correct, n = 0, 0, 0
        all_probs, all_labels  = [], []
        with torch.set_grad_enabled(train):
            for seq, mask, feats, labels, redshifts in loader:
                seq, mask     = seq.to(DEVICE),   mask.to(DEVICE)
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                logits = model(seq, mask, feats)
                if USE_REDSHIFT_WEIGHTING and train:
                    loss = criterion(logits, labels, redshifts=redshifts)
                else:
                    loss = criterion(logits, labels)
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
        ll = log_loss(all_labels, all_probs, labels=list(range(n_classes)))
        return total_loss / n, correct / n, ll

    best_val  = float("inf")
    patience  = 0
    best_probs = None

    label = f"Fold {fold}" if fold is not None else "Final"
    print(f"\nEpoch | Train loss | Train acc | Val acc | Val logloss | Best")
    print("-" * 65)

    for epoch in range(EPOCHS):
        tr_loss, tr_acc, tr_ll = run_epoch(train_loader, train=True)
        va_loss, va_acc, va_ll = run_epoch(val_loader,   train=False)
        scheduler.step()

        improved = va_ll < best_val
        if improved:
            best_val = va_ll
            patience = 0
            ckpt = os.path.join(CHECKPOINT_DIR,
                                f"fold_{fold}.pt" if fold is not None
                                else "plasticc_hybrid_augmented.pt")
            torch.save(model.state_dict(), ckpt)
            # Store val probs for OOF
            model.eval()
            probs_list = []
            with torch.no_grad():
                for seq, mask, feats, labels, _ in val_loader:
                    logits = model(seq.to(DEVICE), mask.to(DEVICE),
                                   feats.to(DEVICE))
                    probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            best_probs = np.vstack(probs_list)
        else:
            patience += 1

        marker = " ✓" if improved else f" ({patience}/{EARLY_STOP_PATIENCE})"
        print(f"{epoch+1:5d} | {tr_loss:.4f}     | {tr_acc:.3f}     | "
              f"{va_acc:.3f}   | {va_ll:.4f}      |{marker}")

        if patience >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"{label} best val log-loss: {best_val:.4f}")
    return best_val, best_probs

# ── Run 5-fold CV ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"5-FOLD CROSS-VALIDATION (split by original object ID)")
print(f"{'='*65}")

for fold, (orig_train_idx, orig_val_idx) in enumerate(kf.split(unique_orig)):
    print(f"\n{'='*65}")
    print(f"FOLD {fold+1}/{N_FOLDS}")
    print(f"{'='*65}")

    # Map original object splits back to sequence indices
    train_orig = set(unique_orig[orig_train_idx])
    val_orig   = set(unique_orig[orig_val_idx])

    train_idx = np.where([get_original_id(oid) in train_orig
                          for oid in valid_ids])[0]
    val_idx   = np.where([get_original_id(oid) in val_orig
                          for oid in valid_ids])[0]

    print(f"Train: {len(train_idx):,} sequences  "
          f"Val: {len(val_idx):,} sequences")

    fold_ll, fold_probs = train_model(train_idx, val_idx, fold=fold+1)
    fold_scores.append(fold_ll)
    oof_probs[val_idx] = fold_probs

# ── OOF evaluation ────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("CROSS-VALIDATION RESULTS")
print(f"{'='*65}")
print(f"\nFold log-losses: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Mean CV log-loss: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

oof_flat = flat_weighted_logloss(y_valid, oof_probs, n_classes)
oof_sk   = log_loss(y_valid, oof_probs, labels=list(range(n_classes)))
print(f"\nOOF sklearn log-loss:       {oof_sk:.4f}")
print(f"OOF flat-weighted log-loss: {oof_flat:.4f}  (Boone: 0.468)")

np.save(os.path.join(CHECKPOINT_DIR, "oof_probs.npy"),  oof_probs)
np.save(os.path.join(CHECKPOINT_DIR, "oof_labels.npy"), y_valid)

# ── Final model: train on ALL data ───────────────────────────────────────────
print(f"\n{'='*65}")
print("TRAINING FINAL MODEL ON ALL DATA")
print(f"{'='*65}")
all_idx = np.arange(len(valid_ids))
train_model(all_idx, all_idx, fold=None)  # val=train for monitoring only

print(f"\nDone! Final checkpoint: {CHECKPOINT_HYBRID}")