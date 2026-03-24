import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import avocado

# ── Constants ─────────────────────────────────────────────────────────────────
BAND_MAP = {'lsstu': 0, 'lsstg': 1, 'lsstr': 2, 'lssti': 3, 'lsstz': 4, 'lssty': 5}
SEQ_LEN  = 350
NUM_CHUNKS       = 8
NUM_TRAIN_CHUNKS = 5

# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_observations(obs, metadata):
    """
    On-the-fly augmentation of a single object's observations.
    - Random flux scaling (simulates different brightness/distance)
    - Random time shift (simulates different observation window)
    - Random noise injection
    - Random observation dropout (simulates sparse cadence)
    """
    obs = obs.copy()

    # 1. Flux scaling — simulate redshift/distance variation (±20%)
    scale = np.random.uniform(0.8, 1.2)
    obs["flux"]       = obs["flux"] * scale
    obs["flux_error"] = obs["flux_error"] * scale

    # 2. Time shift — random offset up to 50 days
    obs["time"] = obs["time"] + np.random.uniform(-50, 50)

    # 3. Add Gaussian noise proportional to flux_error
    noise = np.random.normal(0, obs["flux_error"].values)
    obs["flux"] = obs["flux"] + noise

    # 4. Random observation dropout — drop up to 20% of timesteps
    keep = np.random.rand(len(obs)) > 0.2
    if keep.sum() > 10:   # always keep at least 10 observations
        obs = obs[keep]

    return obs

def obs_to_sequence(obs, augment=False, metadata=None):
    """
    Convert a single object's observations DataFrame to a fixed-length tensor.
    Returns: float32 tensor of shape [SEQ_LEN, 5]
    Features per timestep: [time_norm, flux_norm, flux_err_norm, detected, band_id]
    """
    if augment and metadata is not None:
        obs = augment_observations(obs, metadata)

    obs = obs.sort_values("time")

    # Normalize time to [0, 1]
    t = obs["time"].values.astype(np.float32)
    t_min, t_max = t.min(), t.max()
    t_norm = (t - t_min) / (t_max - t_min + 1e-8)

    # Normalize flux by std (robust to scale differences)
    flux     = obs["flux"].values.astype(np.float32)
    flux_err = obs["flux_error"].values.astype(np.float32)
    flux_std = flux.std() + 1e-8
    flux_norm    = flux / flux_std
    flux_err_norm = flux_err / flux_std

    detected = obs["detected"].values.astype(np.float32)
    band_ids = obs["band"].map(BAND_MAP).fillna(0).values.astype(np.float32)

    # Stack into [T, 5]
    seq = np.stack([t_norm, flux_norm, flux_err_norm, detected, band_ids], axis=1)

    # Pad or truncate to SEQ_LEN
    T = len(seq)
    if T >= SEQ_LEN:
        seq = seq[:SEQ_LEN]
        mask = np.ones(SEQ_LEN, dtype=np.float32)
    else:
        pad  = np.zeros((SEQ_LEN - T, 5), dtype=np.float32)
        seq  = np.vstack([seq, pad])
        mask = np.array([1.0] * T + [0.0] * (SEQ_LEN - T), dtype=np.float32)

    return seq, mask  # [SEQ_LEN, 5], [SEQ_LEN]


class PlasticcDataset(Dataset):
    """
    PyTorch Dataset that returns:
      - sequence tensor [SEQ_LEN, 5]
      - padding mask    [SEQ_LEN]   (1=real, 0=pad)
      - avocado features [n_features]
      - label (int)
    """
    def __init__(self, object_ids, observations_dict, features, labels, augment=False):
        """
        Args:
            object_ids:        list of object_id strings
            observations_dict: dict {object_id: observations_df}
            features:          np.float32 array [N, n_features]
            labels:            np.int64 array [N]
            augment:           whether to apply on-the-fly augmentation
        """
        self.object_ids       = object_ids
        self.observations_dict = observations_dict
        self.features         = features
        self.labels           = labels
        self.augment          = augment

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, idx):
        oid  = self.object_ids[idx]
        obs  = self.observations_dict[oid]
        seq, mask = obs_to_sequence(obs, augment=self.augment)

        return (
            torch.tensor(seq,                  dtype=torch.float32),
            torch.tensor(mask,                 dtype=torch.float32),
            torch.tensor(self.features[idx],   dtype=torch.float32),
            torch.tensor(self.labels[idx],     dtype=torch.long),
        )


def load_observations(num_chunks=NUM_TRAIN_CHUNKS, total_chunks=NUM_CHUNKS):
    """
    Load raw observations for all training chunks.
    Returns dict {object_id: observations_df}
    """
    print("Loading raw observations...")
    obs_dict = {}
    for chunk in range(num_chunks):
        d = avocado.load("plasticc_train", chunk=chunk, num_chunks=total_chunks)
        for obj in d.objects:
            obs_dict[obj.metadata["object_id"]] = obj.observations
        print(f"  Chunk {chunk+1}/{num_chunks}: {len(obs_dict)} objects loaded")
    return obs_dict