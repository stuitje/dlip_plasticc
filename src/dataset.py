import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.cosmology import FlatLambdaCDM
import avocado
import pickle
import os

from config import BAND_MAP, SEQ_LEN, NUM_CHUNKS, NUM_TRAIN_CHUNKS, SCRATCH_DIR

# ── Cosmology ─────────────────────────────────────────────────────────────────
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
_Z_GRID       = np.linspace(0.001, 4.0, 2000)
_DISTMOD_GRID = np.array(COSMO.distmod(_Z_GRID).value, dtype=np.float32)

def redshift_to_distmod(z):
    return float(np.interp(z, _Z_GRID, _DISTMOD_GRID))

Z_TRAIN_MEAN = 0.369
Z_TRAIN_STD  = 0.341
Z_TRAIN_MIN  = 0.011
Z_TRAIN_MAX  = 3.443

def sample_redshift():
    while True:
        z = np.random.normal(Z_TRAIN_MEAN, Z_TRAIN_STD)
        if Z_TRAIN_MIN <= z <= Z_TRAIN_MAX:
            return float(z)

# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_observations(obs, metadata):
    obs = obs.copy()
    is_galactic  = metadata.get("galactic", False)
    z_orig       = metadata.get("redshift", 0.0)
    distmod_orig = metadata.get("true_distmod", 0.0)

    if is_galactic or z_orig <= 0:
        scale = np.random.uniform(0.8, 1.2)
        obs["flux"]       = obs["flux"] * scale
        obs["flux_error"] = obs["flux_error"] * scale
    else:
        z_new         = sample_redshift()
        distmod_new   = redshift_to_distmod(z_new)
        flux_scale    = 10 ** ((distmod_orig - distmod_new) / 2.5)
        obs["flux"]       = obs["flux"] * flux_scale
        obs["flux_error"] = obs["flux_error"] * flux_scale
        obs["time"]       = obs["time"] * (1 + z_new) / (1 + z_orig)

    noise = np.random.normal(0, np.abs(obs["flux_error"].values) * 0.1)
    obs["flux"] = obs["flux"] + noise
    keep = np.random.rand(len(obs)) > 0.2
    if keep.sum() > 10:
        obs = obs[keep]
    return obs

def obs_to_sequence(obs, augment=False, metadata=None):
    if augment and metadata is not None:
        obs = augment_observations(obs, metadata)
    obs    = obs.sort_values("time")
    t      = obs["time"].values.astype(np.float32)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)
    flux     = obs["flux"].values.astype(np.float32)
    flux_err = obs["flux_error"].values.astype(np.float32)
    flux_std = flux.std() + 1e-8
    detected = obs["detected"].values.astype(np.float32)
    band_ids = obs["band"].map(BAND_MAP).fillna(0).values.astype(np.float32)
    seq = np.stack([t_norm, flux / flux_std, flux_err / flux_std,
                    detected, band_ids], axis=1)
    T = len(seq)
    if T >= SEQ_LEN:
        return seq[:SEQ_LEN], np.ones(SEQ_LEN, dtype=np.float32)
    pad  = np.zeros((SEQ_LEN - T, 5), dtype=np.float32)
    mask = np.array([1.0] * T + [0.0] * (SEQ_LEN - T), dtype=np.float32)
    return np.vstack([seq, pad]), mask


class PlasticcDataset(Dataset):
    """
    Returns: seq, mask, features, label, redshift
    redshift is used by RedshiftWeightedLoss during training.
    Galactic objects have redshift=0.
    """
    def __init__(self, object_ids, observations_dict, metadata_dict,
                 features, labels, augment=False):
        self.object_ids        = object_ids
        self.observations_dict = observations_dict
        self.metadata_dict     = metadata_dict
        self.features          = features
        self.labels            = labels
        self.augment           = augment

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, idx):
        oid      = self.object_ids[idx]
        obs      = self.observations_dict[oid]
        metadata = self.metadata_dict[oid]
        seq, mask = obs_to_sequence(obs, augment=self.augment,
                                    metadata=metadata if self.augment else None)
        redshift = float(metadata.get("redshift", 0.0))
        return (
            torch.tensor(seq,                dtype=torch.float32),
            torch.tensor(mask,               dtype=torch.float32),
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],   dtype=torch.long),
            torch.tensor(redshift,           dtype=torch.float32),
        )


def load_observations(num_chunks=NUM_TRAIN_CHUNKS, total_chunks=NUM_CHUNKS):
    cache_path = os.path.join(SCRATCH_DIR, f"obs_cache_{num_chunks}of{total_chunks}.pkl")
    if os.path.exists(cache_path):
        print(f"Loading observations from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            obs_dict, meta_dict = pickle.load(f)
        print(f"  Loaded {len(obs_dict)} objects from cache")
        return obs_dict, meta_dict

    print("Loading raw observations (first time — will cache)...")
    obs_dict, meta_dict = {}, {}
    for chunk in range(num_chunks):
        d = avocado.load("plasticc_train", chunk=chunk, num_chunks=total_chunks)
        for obj in d.objects:
            oid = obj.metadata["object_id"]
            obs_dict[oid]  = obj.observations
            meta_dict[oid] = obj.metadata
        print(f"  Chunk {chunk+1}/{num_chunks}: {len(obs_dict)} objects loaded")

    print(f"Saving cache to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump((obs_dict, meta_dict), f)
    print("Cache saved.")
    return obs_dict, meta_dict