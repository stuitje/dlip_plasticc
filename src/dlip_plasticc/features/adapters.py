from __future__ import annotations

import numpy as np

from avocado.features import Featurizer
from avocado.utils import AvocadoException


class ReshapeFeaturizer(Featurizer):
    """Wrap a featurizer and reshape its output for CNNs.

    The wrapped featurizer's ``select_features`` is called and the resulting
    2D array ``(n_samples, n_features)`` is reshaped to
    ``(n_samples, seq_len, channels)``.

    If ``seq_len`` is not provided, the features are returned as
    ``(n_samples, n_features, 1)``.
    """

    def __init__(self, base_featurizer, seq_len=None, channels=1):
        self.base = base_featurizer
        self.seq_len = seq_len
        self.channels = channels

    def extract_raw_features(self, astronomical_object, return_model=False):
        return self.base.extract_raw_features(
            astronomical_object,
            return_model=return_model,
        )

    def select_features(self, raw_features):
        feats = self.base.select_features(raw_features)

        if hasattr(feats, "values"):
            arr = feats.values.astype(float)
        else:
            arr = np.asarray(feats, dtype=float)

        if arr.ndim != 2:
            raise AvocadoException("Wrapped featurizer must return 2D features.")

        n, f = arr.shape

        if self.seq_len is None:
            seq_len = f
            channels = 1
        else:
            seq_len = self.seq_len
            channels = self.channels

        if seq_len * channels != f:
            raise AvocadoException(
                "Can't reshape features (features=%d) into "
                "(seq_len=%d, channels=%d)" % (f, seq_len, channels)
            )

        arr = arr.reshape(n, seq_len, channels).astype(np.float32)
        return arr