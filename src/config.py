"""
config.py — central configuration for the PLAsTiCC hybrid classifier.
Edit the paths and hyperparameters here before running any scripts.
"""
import os
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
# Root of the scratch/data directory
SCRATCH_DIR = "/scratch/s4339150/plasticc"

# Avocado HDF5 data
DATA_DIR         = os.path.join(SCRATCH_DIR, "data")

# Features (set FEATURES_TAG to switch between v1/v2/etc.)
FEATURES_TAG     = "features_v2"
FEATURES_DIR     = os.path.join(SCRATCH_DIR, "features")
FEATURES_PATH    = os.path.join(FEATURES_DIR, f"{FEATURES_TAG}_plasticc_train.h5")

# Checkpoints and logs (relative to repo root, i.e. dlip_plasticc/)
CHECKPOINT_DIR   = "checkpoints"
LOG_DIR          = "logs"

# Avocado settings file (must be in cwd when running scripts)
AVOCADO_SETTINGS = "avocado_settings.json"

# ── Data config ───────────────────────────────────────────────────────────────
NUM_CHUNKS        = 8    # total chunks the dataset is split into
NUM_TRAIN_CHUNKS  = 8    # how many chunks to use for training (set to 5 for quick runs)

# ── Sequence config ───────────────────────────────────────────────────────────
SEQ_LEN  = 350           # pad/truncate all light curves to this length
BAND_MAP = {             # LSST passband → integer id
    'lsstu': 0, 'lsstg': 1, 'lsstr': 2,
    'lssti': 3, 'lsstz': 4, 'lssty': 5
}

# ── Transformer config ────────────────────────────────────────────────────────
D_MODEL  = 64            # transformer embedding dimension
N_HEADS  = 4             # number of attention heads
N_LAYERS = 2             # number of transformer encoder layers
FFN_DIM  = 128           # feedforward dimension inside transformer

# ── Training config ───────────────────────────────────────────────────────────
BATCH_SIZE           = 64
EPOCHS               = 200
LR                   = 5e-4
WEIGHT_DECAY         = 1e-4
DROPOUT              = 0.4
EARLY_STOP_PATIENCE  = 20
FOCAL_GAMMA          = 2.0

# ── Checkpoint names ──────────────────────────────────────────────────────────
CHECKPOINT_HYBRID    = os.path.join(CHECKPOINT_DIR, "plasticc_hybrid.pt")
CHECKPOINT_MLP_ONLY  = os.path.join(CHECKPOINT_DIR, "plasticc_nn.pt")

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")