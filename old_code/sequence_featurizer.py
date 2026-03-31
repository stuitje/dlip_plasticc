import numpy as np
import pandas as pd

from avocado.features import Featurizer
from avocado.utils import AvocadoException

BAND_MAP = {
    "lsstu": 0,
    "lsstg": 1,
    "lsstr": 2,
    "lssti": 3,
    "lsstz": 4,
    "lssty": 5,
}


class PlasticcSequenceFeaturizer(Featurizer):
    """
    Convert each light curve into a padded event sequence.

    Output:
        features: (n_samples, seq_len, 5)
        mask:     (n_samples, seq_len)

    Channels:
        0: normalized time
        1: normalized flux
        2: normalized flux error
        3: detected
        4: band id
    """

    def __init__(self, seq_len=350, band_map=None):
        self.seq_len = seq_len
        self.band_map = BAND_MAP if band_map is None else band_map

    def _object_to_sequence(self, astronomical_object):
        obs = astronomical_object.observations.copy()

        required_cols = ["time", "band", "flux", "flux_error"]
        for col in required_cols:
            if col not in obs.columns:
                raise AvocadoException(f"Missing required observation column: {col}")

        if "detected" not in obs.columns:
            obs["detected"] = 1.0

        obs = obs.sort_values("time")

        t = obs["time"].values.astype(np.float32)
        flux = obs["flux"].values.astype(np.float32)
        flux_err = obs["flux_error"].values.astype(np.float32)
        detected = obs["detected"].values.astype(np.float32)
        band_ids = obs["band"].map(self.band_map).fillna(0).values.astype(np.float32)

        if len(t) == 0:
            seq = np.zeros((self.seq_len, 5), dtype=np.float32)
            mask = np.zeros(self.seq_len, dtype=np.float32)
            return seq, mask

        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)
        flux_std = np.std(flux) + 1e-8

        seq = np.stack(
            [
                t_norm,
                flux / flux_std,
                flux_err / flux_std,
                detected,
                band_ids,
            ],
            axis=1,
        ).astype(np.float32)

        T = len(seq)
        if T >= self.seq_len:
            seq = seq[: self.seq_len]
            mask = np.ones(self.seq_len, dtype=np.float32)
        else:
            pad = np.zeros((self.seq_len - T, 5), dtype=np.float32)
            seq = np.vstack([seq, pad])
            mask = np.concatenate(
                [np.ones(T, dtype=np.float32), np.zeros(self.seq_len - T, dtype=np.float32)]
            )

        return seq, mask

    def extract_raw_features(self, astronomical_object, return_model=False):
        seq, mask = self._object_to_sequence(astronomical_object)
        raw = {
            "sequence": seq,
            "mask": mask,
        }
        if return_model:
            return raw, None
        return raw

    def select_features(self, raw_features):
        """
        Accepts a DataFrame of object-wise raw features and returns:
            (features, mask)
        """
        if isinstance(raw_features, dict):
            return raw_features["sequence"], raw_features["mask"]

        if isinstance(raw_features, pd.DataFrame):
            if "sequence" not in raw_features.columns or "mask" not in raw_features.columns:
                raise AvocadoException(
                    "Expected raw_features DataFrame to contain 'sequence' and 'mask'."
                )

            features = np.stack(raw_features["sequence"].values).astype(np.float32)
            mask = np.stack(raw_features["mask"].values).astype(np.float32)
            return features, mask

        raise AvocadoException("Unsupported raw_features type in PlasticcSequenceFeaturizer.")