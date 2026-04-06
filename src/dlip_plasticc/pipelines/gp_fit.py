from __future__ import annotations

import numpy as np
from avocado.features import Featurizer
from avocado.plasticc import plasticc_bands


class PlasticcGPGridSequenceFeaturizer(Featurizer):
    def __init__(self, n_timesteps=200):
        self.n_timesteps = n_timesteps

    def extract_raw_features(self, astronomical_object, return_model=False):
        gp, gp_observations, _ = astronomical_object.fit_gaussian_process()

        times = gp_observations["time"].values
        t_min = times.min()
        t_max = times.max()

        grid_times = np.linspace(t_min, t_max, self.n_timesteps, dtype=np.float32)
        duration = np.float32(t_max - t_min)

        gp_fluxes, gp_vars = astronomical_object.predict_gaussian_process(
            plasticc_bands,
            grid_times,
            uncertainties=True,
            fitted_gp=gp,
        )

        return {
            "grid_time": grid_times,                          # (T,)
            "duration": duration,                            # scalar
            "flux": gp_fluxes.T.astype(np.float32),          # (T, 6)
            "sigma": np.sqrt(gp_vars).T.astype(np.float32),  # (T, 6)
        }

    def select_features(self, raw_features):
        flux = np.stack(raw_features["flux"].values)              # (N, T, 6)
        sigma = np.stack(raw_features["sigma"].values)            # (N, T, 6)
        grid_time = np.stack(raw_features["grid_time"].values)    # (N, T)
        duration = np.asarray(raw_features["duration"].values, dtype=np.float32)

        t0 = grid_time[:, :1]
        t1 = grid_time[:, -1:]
        time_norm = (grid_time - t0) / np.clip(t1 - t0, 1e-8, None)
        time_norm = time_norm[:, :, None].astype(np.float32)      # (N, T, 1)

        cont_features = np.concatenate(
            [time_norm, flux.astype(np.float32), sigma.astype(np.float32)],
            axis=2,
        )  # (N, T, 13)

        band_ids = np.zeros((cont_features.shape[0], cont_features.shape[1]), dtype=np.int64)
        mask = np.ones((cont_features.shape[0], cont_features.shape[1]), dtype=np.float32)
        global_features = duration[:, None].astype(np.float32)    # (N, 1)

        return cont_features, band_ids, mask, global_features