from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd
import avocado

from avocado.utils import AvocadoException


MetadataSource = Union[str, pd.DataFrame]


def load_metadata(metadata_source: MetadataSource) -> pd.DataFrame:
    """Load metadata from a dataset name or return it directly."""
    if isinstance(metadata_source, pd.DataFrame):
        metadata = metadata_source.copy()
    else:
        dataset = avocado.load(metadata_source, metadata_only=True)
        metadata = dataset.metadata.copy()

    return metadata


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize prediction rows so each row sums to 1."""
    df = df.copy()
    row_sums = df.sum(axis=1)

    if (row_sums <= 0).any():
        bad = row_sums[row_sums <= 0]
        raise AvocadoException(
            f"Cannot normalize predictions: found {len(bad)} rows with non-positive sums."
        )

    df = df.div(row_sums, axis=0)

    if df.isnull().values.any():
        raise AvocadoException("NaNs encountered while normalizing prediction rows.")

    return df


def _coerce_prediction_columns(
    pred_df: pd.DataFrame,
    true_classes: pd.Series,
) -> pd.DataFrame:
    """Try to coerce prediction columns to the dtype of the true classes."""
    pred_df = pred_df.copy()

    try:
        pred_df.columns = pred_df.columns.astype(true_classes.dtype)
    except Exception:
        pass

    return pred_df


def align_truth_and_predictions(
    metadata_source: MetadataSource,
    pred_df: pd.DataFrame,
    known_classes_only: bool = True,
    normalize: bool = False,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Align ground-truth classes and predictions on common object IDs.

    Parameters
    ----------
    metadata_source
        Either a dataset name understood by `avocado.load(..., metadata_only=True)`
        or a metadata DataFrame indexed by object_id.
    pred_df
        Prediction DataFrame indexed by object_id and columns keyed by class label.
    known_classes_only
        If True, keep only rows whose true class is present in `pred_df.columns`.
    normalize
        If True, row-normalize the predictions before returning.

    Returns
    -------
    true_classes : pandas.Series
    pred_aligned : pandas.DataFrame
    """
    metadata = load_metadata(metadata_source)

    if "class" not in metadata.columns:
        raise AvocadoException("Metadata does not contain a 'class' column.")

    if pred_df.index.name != "object_id":
        pred_df = pred_df.copy()
        pred_df.index.name = "object_id"

    common_idx = metadata.index.intersection(pred_df.index)
    if len(common_idx) == 0:
        raise AvocadoException("No overlap between metadata and predictions.")

    true_classes = metadata.loc[common_idx, "class"].copy()
    pred_aligned = pred_df.loc[common_idx].copy()

    pred_aligned = _coerce_prediction_columns(pred_aligned, true_classes)

    if known_classes_only:
        known_mask = true_classes.isin(set(pred_aligned.columns))
        true_classes = true_classes.loc[known_mask]
        pred_aligned = pred_aligned.loc[known_mask]

    pred_aligned = pred_aligned.reindex(true_classes.index)

    if pred_aligned.isnull().values.any():
        raise AvocadoException("Predictions contain NaNs after alignment.")

    if normalize:
        pred_aligned = normalize_rows(pred_aligned)

    return true_classes, pred_aligned


def score_flat(
    metadata_source: MetadataSource,
    pred_df: pd.DataFrame,
    class_weights: dict | None = None,
    known_classes_only: bool = True,
    normalize: bool = True,
) -> Tuple[float, int]:
    """Compute avocado flat-weighted logloss.

    By default, uses the PLAsTiCC flat weights.
    """
    if class_weights is None:
        class_weights = avocado.plasticc.plasticc_flat_weights

    true_classes, pred_aligned = align_truth_and_predictions(
        metadata_source=metadata_source,
        pred_df=pred_df,
        known_classes_only=known_classes_only,
        normalize=normalize,
    )

    score = avocado.weighted_multi_logloss(
        true_classes,
        pred_aligned,
        class_weights=class_weights,
    )

    return score, len(true_classes)

def score_redshift(
    metadata_source: MetadataSource,
    pred_df: pd.DataFrame,
    class_weights: dict | None = None,
    known_classes_only: bool = True,
    normalize: bool = True,
    redshift_key: str = "redshift",
) -> Tuple[float, int]:
    """Compute avocado redshift-weighted logloss."""
    if class_weights is None:
        class_weights = avocado.plasticc.plasticc_flat_weights

    metadata = load_metadata(metadata_source)

    # Build a temporary dataset-like object for evaluate_weights_redshift
    class _MetadataDataset:
        def __init__(self, metadata):
            self.metadata = metadata

    true_classes, pred_aligned = align_truth_and_predictions(
        metadata_source=metadata,
        pred_df=pred_df,
        known_classes_only=known_classes_only,
        normalize=normalize,
    )

    dataset = _MetadataDataset(metadata.loc[true_classes.index])

    object_weights = avocado.evaluate_weights_redshift(
        dataset, redshift_key=redshift_key
    )

    score = avocado.weighted_multi_logloss(
        true_classes,
        pred_aligned,
        object_weights=object_weights,
        class_weights=class_weights,
    )

    return score, len(true_classes)


def score_kaggle(
    metadata_source: MetadataSource,
    pred_df: pd.DataFrame,
    class_weights: dict | None = None,
    known_classes_only: bool = True,
    normalize: bool = True,
) -> Tuple[float, int]:
    """Compute avocado Kaggle-weighted logloss (no object weights, kaggle class weights)."""
    if class_weights is None:
        class_weights = avocado.plasticc.plasticc_kaggle_weights

    true_classes, pred_aligned = align_truth_and_predictions(
        metadata_source=metadata_source,
        pred_df=pred_df,
        known_classes_only=known_classes_only,
        normalize=normalize,
    )

    score = avocado.weighted_multi_logloss(
        true_classes,
        pred_aligned,
        class_weights=class_weights,
    )

    return score, len(true_classes)