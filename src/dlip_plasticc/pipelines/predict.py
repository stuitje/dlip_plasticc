from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
from tqdm import tqdm
import avocado

from avocado.utils import AvocadoException


def ensure_out_dir(out_dir: str | Path | None) -> Path | None:
    """Create output directory if provided."""
    if out_dir is None:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_chunk_predictions(preds: pd.DataFrame, out_path: str | Path) -> Path:
    """Save one chunk of predictions to HDF5."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preds = preds.copy()
    preds.index.name = "object_id"
    preds.to_hdf(out_path, key="predictions", mode="w")
    return out_path


def combine_predictions(
    prediction_frames: Sequence[pd.DataFrame],
    drop_duplicate_indices: bool = True,
    sort_index: bool = True,
) -> pd.DataFrame:
    """Combine multiple prediction DataFrames into one."""
    if len(prediction_frames) == 0:
        raise AvocadoException("No prediction frames were provided.")

    combined = pd.concat(prediction_frames, axis=0)

    if drop_duplicate_indices:
        combined = combined[~combined.index.duplicated(keep="first")]

    if sort_index:
        combined = combined.sort_index()

    combined.index.name = "object_id"

    if combined.isnull().values.any():
        raise AvocadoException("Combined predictions contain NaNs.")

    return combined


def save_combined_predictions(
    prediction_frames: Sequence[pd.DataFrame],
    out_path: str | Path,
) -> Path:
    """Combine predictions and write them to a single HDF5 file."""
    combined = combine_predictions(prediction_frames)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_hdf(out_path, key="predictions", mode="w")
    return out_path


def load_feature_chunk(
    feature_base: str | Path,
    feature_pattern: str,
    chunk: int,
    feature_key: str = "raw_features",
) -> pd.DataFrame:
    """Load a precomputed raw-feature chunk from HDF5."""
    feature_base = Path(feature_base)
    path = feature_base / feature_pattern.format(chunk=chunk)

    if not path.exists():
        raise FileNotFoundError(f"Missing feature file for chunk {chunk}: {path}")

    df = pd.read_hdf(path, key=feature_key)
    df.index.name = "object_id"
    return df


def predict_partial_from_feature_chunks(
    classifier,
    dataset_name: str,
    total_chunks: int,
    chunks: Iterable[int],
    feature_base: str | Path,
    feature_pattern: str = "features_test_chunk_{chunk}_plasticc_test.h5",
    feature_key: str = "raw_features",
    out_dir: str | Path | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, list[int]]:
    """Predict over dataset chunks using precomputed raw-feature files.

    This is the reusable version of your old `predict_and_score_partial.py`.
    NOTE: only suitable for classifiers that do NOT need dataset.objects
    (i.e. not HybridTransformerClassifier). Use predict_hybrid_partial for that.
    """
    out_dir = ensure_out_dir(out_dir)

    all_preds: List[pd.DataFrame] = []
    processed_chunks: list[int] = []

    iterator = chunks
    if show_progress:
        iterator = tqdm(list(chunks), desc="Chunks", dynamic_ncols=True)

    for chunk in iterator:
        dataset = avocado.load(
            dataset_name,
            chunk=chunk,
            num_chunks=total_chunks,
            metadata_only=True,
        )

        raw_features = load_feature_chunk(
            feature_base=feature_base,
            feature_pattern=feature_pattern,
            chunk=chunk,
            feature_key=feature_key,
        )

        common_idx = dataset.metadata.index.intersection(raw_features.index)
        if len(common_idx) == 0:
            continue

        dataset.metadata = dataset.metadata.loc[common_idx].copy()
        dataset.raw_features = raw_features.loc[common_idx].copy()

        preds = classifier.predict(dataset)
        preds.index.name = "object_id"

        if not preds.index.equals(dataset.metadata.index):
            preds = preds.reindex(dataset.metadata.index)

        if preds.isnull().values.any():
            raise AvocadoException(f"Predictions contain NaNs for chunk {chunk}.")

        if out_dir is not None:
            save_chunk_predictions(preds, out_dir / f"predictions_chunk_{chunk}.h5")

        all_preds.append(preds)
        processed_chunks.append(chunk)

    if len(all_preds) == 0:
        raise AvocadoException("No predictions were generated.")

    combined = combine_predictions(all_preds)

    if out_dir is not None:
        combined.to_hdf(out_dir / "predictions_combined.h5", key="predictions", mode="w")

    return combined, processed_chunks


def predict_hybrid_partial(
    classifier,
    dataset_name: str,
    total_chunks: int,
    chunks: Iterable[int],
    feature_base: str | Path,
    feature_pattern: str = "features_test_chunk_{chunk}_plasticc_test.h5",
    feature_key: str = "raw_features",
    out_dir: str | Path | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, list[int]]:
    """Predict over dataset chunks for HybridTransformerClassifier.

    Unlike predict_partial_from_feature_chunks, this function loads full
    observations (metadata_only=False) so that the sequence branch of the
    hybrid model has access to dataset.objects. Precomputed GP feature chunks
    are still injected via dataset.raw_features to avoid recomputing them.

    Parameters
    ----------
    classifier
        A trained HybridTransformerClassifier instance.
    dataset_name
        Avocado dataset name (e.g. "plasticc_test").
    total_chunks
        Total number of chunks the dataset is split into.
    chunks
        Iterable of chunk indices to predict over.
    feature_base
        Directory containing precomputed GP feature HDF5 files.
    feature_pattern
        Filename pattern for feature chunks, with a {chunk} placeholder.
    feature_key
        HDF5 key for the raw features table.
    out_dir
        If provided, per-chunk and combined predictions are saved here.
    show_progress
        Whether to show a tqdm progress bar.
    """
    out_dir = ensure_out_dir(out_dir)

    all_preds: List[pd.DataFrame] = []
    processed_chunks: list[int] = []

    iterator = chunks
    if show_progress:
        iterator = tqdm(list(chunks), desc="Chunks", dynamic_ncols=True)

    for chunk in iterator:
        # Load with observations so dataset.objects is populated for the
        # sequence branch. This is the critical difference from
        # predict_partial_from_feature_chunks.
        dataset = avocado.load(
            dataset_name,
            chunk=chunk,
            num_chunks=total_chunks,
            metadata_only=False,
        )

        # Inject precomputed GP features so the tabular branch does not need
        # to recompute them from scratch.
        raw_features = load_feature_chunk(
            feature_base=feature_base,
            feature_pattern=feature_pattern,
            chunk=chunk,
            feature_key=feature_key,
        )

        common_idx = dataset.metadata.index.intersection(raw_features.index)
        if len(common_idx) == 0:
            continue

        dataset.metadata = dataset.metadata.loc[common_idx].copy()
        dataset.raw_features = raw_features.loc[common_idx].copy()

        # Filter dataset.objects to match common_idx so the sequence branch
        # iterates over exactly the same objects as the tabular branch.
        if dataset.objects is not None:
            dataset.objects = [
                obj for obj in dataset.objects
                if obj.metadata["object_id"] in common_idx
            ]

        preds = classifier.predict(dataset)
        preds.index.name = "object_id"

        if not preds.index.equals(dataset.metadata.index):
            preds = preds.reindex(dataset.metadata.index)

        if preds.isnull().values.any():
            raise AvocadoException(f"Predictions contain NaNs for chunk {chunk}.")

        if out_dir is not None:
            save_chunk_predictions(preds, out_dir / f"predictions_chunk_{chunk}.h5")

        all_preds.append(preds)
        processed_chunks.append(chunk)

    if len(all_preds) == 0:
        raise AvocadoException("No predictions were generated.")

    combined = combine_predictions(all_preds)

    if out_dir is not None:
        combined.to_hdf(out_dir / "predictions_combined.h5", key="predictions", mode="w")

    return combined, processed_chunks


def predict_partial_from_dataset(
    classifier,
    featurizer,
    dataset_name: str,
    total_chunks: int,
    chunks: Iterable[int],
    out_dir: str | Path | None = None,
    metadata_only: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, list[int]]:
    """Predict over dataset chunks with online feature extraction.

    This is the reusable version of your old `predict_transformer_partial.py`.
    """
    out_dir = ensure_out_dir(out_dir)

    all_preds: List[pd.DataFrame] = []
    processed_chunks: list[int] = []

    iterator = chunks
    if show_progress:
        iterator = tqdm(list(chunks), desc="Chunks", dynamic_ncols=True)

    for chunk in iterator:
        dataset = avocado.load(
            dataset_name,
            chunk=chunk,
            num_chunks=total_chunks,
            metadata_only=metadata_only,
        )

        if dataset.objects is None:
            raise AvocadoException(
                "Online prediction requires observations to be loaded, "
                "but this dataset chunk was metadata-only."
            )

        dataset.extract_raw_features(featurizer)
        preds = classifier.predict(dataset)
        preds.index.name = "object_id"

        if preds.isnull().values.any():
            raise AvocadoException(f"Predictions contain NaNs for chunk {chunk}.")

        if out_dir is not None:
            save_chunk_predictions(preds, out_dir / f"predictions_chunk_{chunk}.h5")

        all_preds.append(preds)
        processed_chunks.append(chunk)

    if len(all_preds) == 0:
        raise AvocadoException("No predictions were generated.")

    combined = combine_predictions(all_preds)

    if out_dir is not None:
        combined.to_hdf(out_dir / "predictions_combined.h5", key="predictions", mode="w")

    return combined, processed_chunks