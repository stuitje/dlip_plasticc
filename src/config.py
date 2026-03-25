"""
config.py — central configuration for the PLAsTiCC hybrid classifier.
Edit the paths and hyperparameters here before running any scripts.
"""
import os
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH_DIR      = "/scratch/s4339150/plasticc"
DATA_DIR         = os.path.join(SCRATCH_DIR, "data")
FEATURES_TAG     = "features_v2"
FEATURES_DIR     = os.path.join(SCRATCH_DIR, "features")
FEATURES_PATH    = os.path.join(FEATURES_DIR, f"{FEATURES_TAG}_plasticc_train.h5")
CHECKPOINT_DIR   = "checkpoints"
LOG_DIR          = "logs"

# ── Data config ───────────────────────────────────────────────────────────────
NUM_CHUNKS        = 8
NUM_TRAIN_CHUNKS  = 8

# ── Sequence config ───────────────────────────────────────────────────────────
SEQ_LEN  = 350
BAND_MAP = {
    'lsstu': 0, 'lsstg': 1, 'lsstr': 2,
    'lssti': 3, 'lsstz': 4, 'lssty': 5
}

# ── Transformer config ────────────────────────────────────────────────────────
D_MODEL  = 64
N_HEADS  = 4
N_LAYERS = 2
FFN_DIM  = 128

# ── Training config ───────────────────────────────────────────────────────────
BATCH_SIZE           = 64
EPOCHS               = 200
LR                   = 5e-4
WEIGHT_DECAY         = 1e-4
DROPOUT              = 0.4
EARLY_STOP_PATIENCE  = 20
FOCAL_GAMMA          = 2.0

# ── Redshift weighting ────────────────────────────────────────────────────────
USE_REDSHIFT_WEIGHTING = False  # set False to disable
REDSHIFT_SCALE         = 0.5   # higher = more weight on high-z objects
                                # tune between 0.5 and 2.0

# ── Checkpoint names ──────────────────────────────────────────────────────────
CHECKPOINT_HYBRID   = os.path.join(CHECKPOINT_DIR, "plasticc_hybrid.pt")
CHECKPOINT_MLP_ONLY = os.path.join(CHECKPOINT_DIR, "plasticc_nn.pt")

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")