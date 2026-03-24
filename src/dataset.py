import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.cosmology import FlatLambdaCDM
import avocado

from config import BAND_MAP, SEQ_LEN, NUM_CHUNKS, NUM_TRAIN_CHUNKS

# ── Cosmology (Planck 2018, same as PLAsTiCC simulation) ─────────────────────
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

# Redshift grid for fast distmod lookup
_Z_GRID      = np.linspace(0.001, 4.0, 2000)
_DISTMOD_GRID = np.array(COSMO.distmod(_Z_GRID).value, dtype=np.float32)

def redshift_to_distmod(z):
    """Fast distmod lookup via interpolation."""
    return float(np.interp(z, _Z_GRID, _DISTMOD_GRID))

# Training redshift distribution for sampling z_new
# Fit a KDE-like empirical distribution from training data
# We use a simple truncated normal mixture as an approximation
Z_TRAIN_MEAN = 0.369
Z_TRAIN_STD  = 0.341
Z_TRAIN_MIN  = 0.011
Z_TRAIN_MAX  = 3.443

def sample_redshift():
    """Sample a new redshift from the training distribution."""
    while True:
        z = np.random.normal(Z_TRAIN_MEAN, Z_TRAIN_STD)
        if Z_TRAIN_MIN <= z <= Z_TRAIN_MAX:
            return float(z)

# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_observations(obs, metadata):
    """
    Physically motivated augmentation.

    Galactic objects (no redshift):
      - Random flux scaling
      - Random noise injection
      - Random observation dropout

    Extragalactic objects:
      - Redshift augmentation: sample z_new, rescale flux via distance modulus,
        dilate time by (1+z_new)/(1+z_orig)
      - Random noise injection
      - Random observation dropout
    """
    obs = obs.copy()
    is_galactic = metadata.get("galactic", False)
    z_orig      = metadata.get("redshift", 0.0)
    distmod_orig = metadata.get("true_distmod", 0.0)

    if is_galactic or z_orig <= 0:
        # ── Galactic: simple augmentation only ───────────────────────────────
        scale = np.random.uniform(0.8, 1.2)
        obs["flux"]       = obs["flux"] * scale
        obs["flux_error"] = obs["flux_error"] * scale
    else:
        # ── Extragalactic: redshift augmentation ──────────────────────────────
        z_new        = sample_redshift()
        distmod_new  = redshift_to_distmod(z_new)

        # Flux scaling: delta_distmod in magnitudes → flux ratio
        delta_distmod = distmod_orig - distmod_new
        flux_scale    = 10 ** (delta_distmod / 2.5)

        obs["flux"]       = obs["flux"] * flux_scale
        obs["flux_error"] = obs["flux_error"] * flux_scale

        # Time dilation
        time_scale = (1 + z_new) / (1 + z_orig)
        obs["time"] = obs["time"] * time_scale

    # ── Common: noise + dropout ───────────────────────────────────────────────
    # Add Gaussian noise proportional to flux_error
    noise = np.random.normal(0, np.abs(obs["flux_error"].values) * 0.1)
    obs["flux"] = obs["flux"] + noise

    # Random observation dropout (drop up to 20%)
    keep = np.random.rand(len(obs)) > 0.2
    if keep.sum() > 10:
        obs = obs[keep]

    return obs


def obs_to_sequence(obs, augment=False, metadata=None):
    """
    Convert observations DataFrame to fixed-length tensor.
    Returns: seq [SEQ_LEN, 5], mask [SEQ_LEN]
    Features: [time_norm, flux_norm, flux_err_norm, detected, band_id]
    """
    if augment and metadata is not None:
        obs = augment_observations(obs, metadata)

    obs = obs.sort_values("time")

    t      = obs["time"].values.astype(np.float32)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)

    flux     = obs["flux"].values.astype(np.float32)
    flux_err = obs["flux_error"].values.astype(np.float32)
    flux_std = flux.std() + 1e-8
    flux_norm     = flux / flux_std
    flux_err_norm = flux_err / flux_std

    detected = obs["detected"].values.astype(np.float32)
    band_ids = obs["band"].map(BAND_MAP).fillna(0).values.astype(np.float32)

    seq = np.stack([t_norm, flux_norm, flux_err_norm, detected, band_ids], axis=1)

    T = len(seq)
    if T >= SEQ_LEN:
        seq  = seq[:SEQ_LEN]
        mask = np.ones(SEQ_LEN, dtype=np.float32)
    else:
        pad  = np.zeros((SEQ_LEN - T, 5), dtype=np.float32)
        seq  = np.vstack([seq, pad])
        mask = np.array([1.0] * T + [0.0] * (SEQ_LEN - T), dtype=np.float32)

    return seq, mask


class PlasticcDataset(Dataset):
    """
    PyTorch Dataset returning:
      seq   [SEQ_LEN, 5]    raw light curve sequence
      mask  [SEQ_LEN]       1=real timestep, 0=padding
      feats [n_features]    avocado handcrafted features
      label int             class index
    """
    def __init__(self, object_ids, observations_dict, metadata_dict,
                 features, labels, augment=False):
        self.object_ids        = object_ids
        self.observations_dict = observations_dict
        self.metadata_dict     = metadata_dict  # needed for redshift augmentation
        self.features          = features
        self.labels            = labels
        self.augment           = augment

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, idx):
        oid      = self.object_ids[idx]
        obs      = self.observations_dict[oid]
        metadata = self.metadata_dict[oid] if self.augment else None
        seq, mask = obs_to_sequence(obs, augment=self.augment, metadata=metadata)
        return (
            torch.tensor(seq,                dtype=torch.float32),
            torch.tensor(mask,               dtype=torch.float32),
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],   dtype=torch.long),
        )


def load_observations(num_chunks=NUM_TRAIN_CHUNKS, total_chunks=NUM_CHUNKS):
    """
    Load raw observations and metadata for all training chunks.
    Returns:
        obs_dict:  {object_id: observations_df}
        meta_dict: {object_id: metadata_dict}
    """
    print("Loading raw observations...")
    obs_dict  = {}
    meta_dict = {}
    for chunk in range(num_chunks):
        d = avocado.load("plasticc_train", chunk=chunk, num_chunks=total_chunks)
        for obj in d.objects:
            oid = obj.metadata["object_id"]
            obs_dict[oid]  = obj.observations
            meta_dict[oid] = obj.metadata
        print(f"  Chunk {chunk+1}/{num_chunks}: {len(obs_dict)} objects loaded")
    return obs_dict, meta_dict