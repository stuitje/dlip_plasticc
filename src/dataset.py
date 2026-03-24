import numpy as np
import torch
from torch.utils.data import Dataset
import avocado

from config import BAND_MAP, SEQ_LEN, NUM_CHUNKS, NUM_TRAIN_CHUNKS

# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_observations(obs, metadata=None):
    obs = obs.copy()
    scale = np.random.uniform(0.8, 1.2)
    obs["flux"]       = obs["flux"] * scale
    obs["flux_error"] = obs["flux_error"] * scale
    obs["time"]       = obs["time"] + np.random.uniform(-50, 50)
    noise = np.random.normal(0, obs["flux_error"].values)
    obs["flux"] = obs["flux"] + noise
    keep = np.random.rand(len(obs)) > 0.2
    if keep.sum() > 10:
        obs = obs[keep]
    return obs

def obs_to_sequence(obs, augment=False, metadata=None):
    if augment and metadata is not None:
        obs = augment_observations(obs, metadata)
    obs = obs.sort_values("time")
    t       = obs["time"].values.astype(np.float32)
    t_norm  = (t - t.min()) / (t.max() - t.min() + 1e-8)
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
    def __init__(self, object_ids, observations_dict, features, labels, augment=False):
        self.object_ids        = object_ids
        self.observations_dict = observations_dict
        self.features          = features
        self.labels            = labels
        self.augment           = augment

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, idx):
        oid      = self.object_ids[idx]
        obs      = self.observations_dict[oid]
        seq, mask = obs_to_sequence(obs, augment=self.augment)
        return (
            torch.tensor(seq,                dtype=torch.float32),
            torch.tensor(mask,               dtype=torch.float32),
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],   dtype=torch.long),
        )

def load_observations(num_chunks=NUM_TRAIN_CHUNKS, total_chunks=NUM_CHUNKS):
    print("Loading raw observations...")
    obs_dict = {}
    for chunk in range(num_chunks):
        d = avocado.load("plasticc_train", chunk=chunk, num_chunks=total_chunks)
        for obj in d.objects:
            obs_dict[obj.metadata["object_id"]] = obj.observations
        print(f"  Chunk {chunk+1}/{num_chunks}: {len(obs_dict)} objects loaded")
    return obs_dict