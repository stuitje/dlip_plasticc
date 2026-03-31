from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from avocado.utils import AvocadoException

from .score import normalize_rows, score_flat


def read_prediction_hdf(path: str | Path, key: str = "predictions") -> pd.DataFrame:
    """Read a predictions HDF file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {path}")

    df = pd.read_hdf(path, key=key)
    df.index.name = "object_id"
    return df


def align_prediction_frames(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    normalize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two prediction frames on common rows and columns."""
    common_idx = pred_a.index.intersection(pred_b.index)
    if len(common_idx) == 0:
        raise AvocadoException("No overlapping object_ids between prediction frames.")

    pred_a = pred_a.loc[common_idx].sort_index()
    pred_b = pred_b.loc[common_idx].sort_index()

    common_cols = pred_a.columns.intersection(pred_b.columns)
    if len(common_cols) == 0:
        raise AvocadoException("No overlapping class columns between prediction frames.")

    pred_a = pred_a[common_cols].copy()
    pred_b = pred_b[common_cols].copy()

    if normalize:
        pred_a = normalize_rows(pred_a)
        pred_b = normalize_rows(pred_b)

    return pred_a, pred_b


def blend_predictions(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    weight_a: float,
    normalize: bool = True,
) -> pd.DataFrame:
    """Blend two aligned prediction frames."""
    if not (0.0 <= weight_a <= 1.0):
        raise AvocadoException("weight_a must be between 0 and 1.")

    pred_a, pred_b = align_prediction_frames(pred_a, pred_b, normalize=normalize)

    blended = weight_a * pred_a + (1.0 - weight_a) * pred_b

    if normalize:
        blended = normalize_rows(blended)

    blended.index.name = "object_id"
    return blended


def grid_search_blend_weight(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    dataset_name: str,
    grid_step: float = 0.05,
) -> tuple[float, float, pd.DataFrame, pd.DataFrame]:
    """Search for the best linear blend weight on flat-weighted logloss.

    Returns
    -------
    best_weight_a : float
    best_score : float
    best_blend : pandas.DataFrame
    history : pandas.DataFrame
    """
    if grid_step <= 0 or grid_step > 1:
        raise AvocadoException("grid_step must be in the interval (0, 1].")

    pred_a, pred_b = align_prediction_frames(pred_a, pred_b, normalize=True)

    best_score = np.inf
    best_weight = None
    best_blend = None
    history_rows = []

    weights = np.arange(0.0, 1.0 + 1e-12, grid_step)

    for w in weights:
        blended = w * pred_a + (1.0 - w) * pred_b
        blended = normalize_rows(blended)

        score, n_scored = score_flat(dataset_name, blended)

        history_rows.append(
            {
                "weight_a": float(w),
                "flat_logloss": float(score),
                "n_scored": int(n_scored),
            }
        )

        if score < best_score:
            best_score = score
            best_weight = float(w)
            best_blend = blended.copy()

    history = pd.DataFrame(history_rows)

    if best_blend is None or best_weight is None:
        raise AvocadoException("Blend search failed to produce a valid result.")

    best_blend.index.name = "object_id"
    return best_weight, float(best_score), best_blend, history


def save_blended_predictions(
    blended: pd.DataFrame,
    out_path: str | Path,
    key: str = "predictions",
) -> Path:
    """Save blended predictions to HDF5."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blended = blended.copy()
    blended.index.name = "object_id"
    blended.to_hdf(out_path, key=key, mode="w")

    return out_path