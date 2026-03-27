import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
import os

from config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, EARLY_STOP_PATIENCE,
    FOCAL_GAMMA, CHECKPOINT_DIR, DEVICE,
    USE_REDSHIFT_WEIGHTING, REDSHIFT_SCALE,
    N_LAYERS, D_MODEL, N_HEADS, FFN_DIM, DROPOUT
)
from utils import HybridPlasticcNet, FocalLoss, RedshiftWeightedLoss, build_dataset, class_weights
from dataset import PlasticcDataset, load_observations

# ── Config ────────────────────────────────────────────────────────────────────
N_MODELS = 5
SEEDS    = [42, 123, 456, 789, 1337]

print(f"Using device: {DEVICE}")
print(f"Training ensemble of {N_MODELS} models")

# ── Data (load once, reuse for all models) ────────────────────────────────────
X, y, le, scaler, object_ids = build_dataset()
n_classes = len(le.classes_)
print(f"Classes ({n_classes}): {le.classes_}")

obs_dict, meta_dict = load_observations()
valid_ids = [oid for oid in object_ids if oid in obs_dict]
valid_idx = [object_ids.index(oid) for oid in valid_ids]
X_valid   = X[valid_idx]
y_valid   = y[valid_idx]
print(f"Objects with observations: {len(valid_ids)}")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Store val predictions from each model
ensemble_probs = []
val_idx_global = None

# ── Train each model ──────────────────────────────────────────────────────────
for model_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*65}")
    print(f"Training model {model_idx+1}/{N_MODELS}  (seed={seed})")
    print(f"{'='*65}")

    # Reproducible split for this seed
    rng      = np.random.default_rng(seed)
    n_total  = len(valid_ids)
    n_val    = int(0.15 * n_total)
    n_train  = n_total - n_val
    idx_perm = rng.permutation(n_total)
    train_idx = idx_perm[:n_train]
    val_idx   = idx_perm[n_train:]

    # Store val indices from first model for consistent evaluation
    if val_idx_global is None:
        val_idx_global = val_idx

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = PlasticcDataset(
        [valid_ids[i] for i in train_idx], obs_dict, meta_dict,
        X_valid[train_idx], y_valid[train_idx], augment=True
    )
    val_ds = PlasticcDataset(
        [valid_ids[i] for i in val_idx_global], obs_dict, meta_dict,
        X_valid[val_idx_global], y_valid[val_idx_global], augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = HybridPlasticcNet(X.shape[1], n_classes).to(DEVICE)
    cw    = class_weights(y_valid, n_classes, DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

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
        logloss    = log_loss(all_labels, all_probs, labels=list(range(n_classes)))
        return total_loss / n, correct / n, logloss

    best_val_logloss = float("inf")
    patience_counter = 0
    best_probs       = None

    print(f"Epoch | Train loss | Train acc | Val acc | Val logloss | Best")
    print("-" * 65)

    for epoch in range(EPOCHS):
        tr_loss, tr_acc, tr_ll = run_epoch(train_loader, train=True)
        va_loss, va_acc, va_ll = run_epoch(val_loader,   train=False)
        scheduler.step()

        improved = va_ll < best_val_logloss
        if improved:
            best_val_logloss = va_ll
            patience_counter = 0
            # Save checkpoint and best probs for this model
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ensemble_{model_idx}.pt")
            torch.save(model.state_dict(), ckpt_path)
            # Store val probs at best epoch
            model.eval()
            probs_list = []
            with torch.no_grad():
                for seq, mask, feats, labels, _ in val_loader:
                    logits = model(seq.to(DEVICE), mask.to(DEVICE), feats.to(DEVICE))
                    probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            best_probs = np.vstack(probs_list)
        else:
            patience_counter += 1

        marker = " ✓" if improved else f" ({patience_counter}/{EARLY_STOP_PATIENCE})"
        print(f"{epoch+1:5d} | {tr_loss:.4f}     | {tr_acc:.3f}     | "
              f"{va_acc:.3f}   | {va_ll:.4f}      |{marker}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"Model {model_idx+1} best val log-loss: {best_val_logloss:.4f}")
    ensemble_probs.append(best_probs)

# ── Ensemble evaluation ───────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("ENSEMBLE RESULTS")
print(f"{'='*65}")

true_labels = y_valid[val_idx_global]

# Average probabilities across all models
avg_probs = np.mean(ensemble_probs, axis=0)
ensemble_logloss = log_loss(true_labels, avg_probs, labels=list(range(n_classes)))
ensemble_acc     = (avg_probs.argmax(axis=1) == true_labels).mean()

print(f"\nIndividual model log-losses:")
for i, probs in enumerate(ensemble_probs):
    ll = log_loss(true_labels, probs, labels=list(range(n_classes)))
    print(f"  Model {i+1}: {ll:.4f}")

print(f"\nEnsemble log-loss : {ensemble_logloss:.4f}")
print(f"Ensemble accuracy : {ensemble_acc:.4f}")

# Save ensemble probabilities
np.save(os.path.join(CHECKPOINT_DIR, "ensemble_probs.npy"), avg_probs)
np.save(os.path.join(CHECKPOINT_DIR, "ensemble_labels.npy"), true_labels)
print(f"\nEnsemble probs saved to {CHECKPOINT_DIR}/ensemble_probs.npy")