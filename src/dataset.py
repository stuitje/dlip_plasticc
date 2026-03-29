import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os

from config import (BAND_MAP, SEQ_LEN, SCRATCH_DIR,
                    OBS_DATASET_NAME, OBS_NUM_CHUNKS, OBS_TOTAL_CHUNKS)

# ── Cached augmented sequences ────────────────────────────────────────────────
AUG_SEQUENCES_PATH = os.path.join(SCRATCH_DIR, "augmented_sequences.pkl")
_aug_seq_cache = None

def get_aug_sequences():
    """Load precomputed augmented sequences cache (from transformer.py)."""
    global _aug_seq_cache
    if _aug_seq_cache is None:
        print(f"Loading augmented sequences cache: {AUG_SEQUENCES_PATH}")
        with open(AUG_SEQUENCES_PATH, "rb") as f:
            data = pickle.load(f)
        # Build {object_id: (seq, mask)} lookup
        _aug_seq_cache = {
            oid: (data["sequences"][i], data["masks"][i])
            for i, oid in enumerate(data["object_ids"])
        }
        print(f"  Loaded sequences for {len(_aug_seq_cache):,} objects")
    return _aug_seq_cache

def get_original_id(oid):
    """Strip augmentation suffix to get original object ID."""
    return oid.split("_aug_")[0] if "_aug_" in oid else oid

# ── Simple sequence conversion (for original objects / test set) ──────────────
def obs_to_sequence(obs, augment=False, metadata=None):
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
    Dataset that uses precomputed augmented sequences for the transformer branch.
    - If object_id is in aug_seq_cache: uses the actual GP-resampled sequence
    - Otherwise: falls back to original observations
    """
    def __init__(self, object_ids, observations_dict, metadata_dict,
                 features, labels, augment=False, use_aug_sequences=True):
        self.object_ids        = object_ids
        self.observations_dict = observations_dict
        self.metadata_dict     = metadata_dict
        self.features          = features
        self.labels            = labels
        self.augment           = augment
        self.use_aug_sequences = use_aug_sequences
        if use_aug_sequences:
            self.aug_seqs = get_aug_sequences()
        else:
            self.aug_seqs = {}

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, idx):
        oid      = self.object_ids[idx]
        orig_oid = get_original_id(oid)
        redshift = 0.0

        # ── Transformer sequence ──────────────────────────────────────────────
        if self.augment and orig_oid in self.observations_dict:
            # On-the-fly redshift augmentation — new random redshift every epoch
            # gives transformer branch infinite variety beyond fixed aug_seqs
            obs      = self.observations_dict[orig_oid]
            metadata = self.metadata_dict[orig_oid]
            seq, mask = obs_to_sequence(obs, augment=True, metadata=metadata)
            redshift  = float(metadata.get("redshift", 0.0))
        elif oid in self.aug_seqs:
            # Use pre-generated GP-resampled augmented sequence
            seq, mask = self.aug_seqs[oid]
        elif orig_oid in self.observations_dict:
            # Fall back to original observations (no augmentation)
            obs      = self.observations_dict[orig_oid]
            metadata = self.metadata_dict[orig_oid]
            seq, mask = obs_to_sequence(obs)
            redshift  = float(metadata.get("redshift", 0.0))
        else:
            # Zero sequence as last resort
            seq  = np.zeros((SEQ_LEN, 5), dtype=np.float32)
            mask = np.zeros(SEQ_LEN,      dtype=np.float32)

        # Get redshift from metadata if available
        if orig_oid in self.metadata_dict:
            redshift = float(self.metadata_dict[orig_oid].get("redshift", 0.0))

        return (
            torch.tensor(seq,                dtype=torch.float32),
            torch.tensor(mask,               dtype=torch.float32),
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],   dtype=torch.long),
            torch.tensor(redshift,           dtype=torch.float32),
        )


def load_observations(num_chunks=OBS_NUM_CHUNKS, total_chunks=OBS_TOTAL_CHUNKS):
    """
    Load original training observations for fallback.
    Augmented sequences are loaded separately via get_aug_sequences().
    """
    import avocado
    cache_path = os.path.join(SCRATCH_DIR,
                               f"obs_cache_{num_chunks}of{total_chunks}.pkl")

    if os.path.exists(cache_path):
        print(f"Loading observations from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            obs_dict, meta_dict = pickle.load(f)
        print(f"  Loaded {len(obs_dict)} objects from cache")
        return obs_dict, meta_dict

    print(f"Loading raw observations from {OBS_DATASET_NAME}...")
    obs_dict, meta_dict = {}, {}
    for chunk in range(num_chunks):
        d = avocado.load(OBS_DATASET_NAME, chunk=chunk,
                         num_chunks=total_chunks)
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