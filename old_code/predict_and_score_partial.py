#!/usr/bin/env python3
"""
Predict and score an avocado classifier on a partial set of PLAsTiCC test chunks.

This avoids `avocado_predict`, which expects all feature chunks to exist and
match the full dataset chunking metadata.

What it does:
- loads the classifier
- loops over available test chunks
- loads the matching raw feature chunk file directly
- attaches raw features to the avocado Dataset
- predicts for that chunk
- saves per-chunk predictions
- combines predictions across chunks
- scores on the overlapping labeled objects only

Usage:
    python predict_and_score_partial.py \
        --classifier my_cnn \
        --dataset plasticc_test \
        --total-chunks 500 \
        --chunks 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
        --feature-base /home6/s4339150/Courses/plasticc_scratch/features \
        --out-dir /scratch/s4339150/plasticc/predictions_partial
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

import avocado


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", required=True, help="Classifier name, e.g. my_cnn")
    parser.add_argument("--dataset", default="plasticc_test", help="Dataset name")
    parser.add_argument("--total-chunks", type=int, default=500, help="Total dataset chunks")
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        required=True,
        help="Available chunk ids to process, e.g. --chunks 0 1 2 3",
    )
    parser.add_argument(
        "--feature-base",
        required=True,
        help="Directory containing features_test_chunk_{k}_plasticc_test.h5 files",
    )
    parser.add_argument(
        "--feature-pattern",
        default="features_test_chunk_{chunk}_plasticc_test.h5",
        help="Filename pattern for feature chunks",
    )
    parser.add_argument(
        "--feature-key",
        default="raw_features",
        help="HDF5 key for raw features",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for chunk predictions and combined file",
    )
    parser.add_argument(
        "--score-flat",
        action="store_true",
        help="Also compute avocado flat-weighted logloss on overlapping labeled objects",
    )
    return parser.parse_args()


def load_feature_chunk(feature_base: str, feature_pattern: str, chunk: int, feature_key: str) -> pd.DataFrame:
    path = os.path.join(feature_base, feature_pattern.format(chunk=chunk))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing feature file for chunk {chunk}: {path}")
    df = pd.read_hdf(path, key=feature_key)
    df.index.name = "object_id"
    return df


def score_predictions(dataset_name: str, total_chunks: int, pred_df: pd.DataFrame):
    """
    Score on overlapping labeled objects only.
    """
    print("\nLoading full test metadata for scoring...")
    dataset_full = avocado.load(dataset_name, metadata_only=True)
    metadata = dataset_full.metadata

    common_idx = metadata.index.intersection(pred_df.index)
    if len(common_idx) == 0:
        raise RuntimeError("No overlap between metadata and predictions for scoring.")

    true_classes = metadata.loc[common_idx, "class"]
    pred_aligned = pred_df.loc[common_idx].copy()

    # Keep only rows whose true class is actually predicted by the classifier
    known_classes = set(pred_aligned.columns)
    known_mask = true_classes.isin(known_classes)

    true_classes = true_classes.loc[known_mask]
    pred_aligned = pred_aligned.loc[known_mask]

    # ensure identical ordering
    pred_aligned = pred_aligned.reindex(true_classes.index)

    # keep only columns that correspond to present prediction classes
    # weighted_multi_logloss expects prediction columns keyed by class labels
    try:
        pred_aligned.columns = pred_aligned.columns.astype(true_classes.dtype)
    except Exception:
        pass

    if pred_aligned.isnull().sum().sum() != 0:
        raise RuntimeError("Predictions contain NaNs after alignment.")

    flat_score = avocado.weighted_multi_logloss(
        true_classes,
        pred_aligned,
        class_weights=avocado.plasticc.plasticc_flat_weights,
    )

    return flat_score, len(common_idx)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading classifier '{args.classifier}'...")
    classifier = avocado.Classifier.load(args.classifier)

    all_preds = []
    processed_chunks = []

    for chunk in tqdm(args.chunks, desc="Chunks", dynamic_ncols=True):
        print(f"\n=== Processing chunk {chunk}/{args.total_chunks - 1} ===")

        # Load dataset chunk metadata only
        dataset = avocado.load(
            args.dataset,
            chunk=chunk,
            num_chunks=args.total_chunks,
            metadata_only=True,
        )

        # Load raw feature chunk directly
        raw_features = load_feature_chunk(
            args.feature_base,
            args.feature_pattern,
            chunk,
            args.feature_key,
        )

        # Align raw features to metadata overlap
        common_idx = dataset.metadata.index.intersection(raw_features.index)
        if len(common_idx) == 0:
            print(f"  No overlap for chunk {chunk}, skipping.")
            continue

        dataset.metadata = dataset.metadata.loc[common_idx].copy()
        dataset.raw_features = raw_features.loc[common_idx].copy()

        # Predict
        preds = classifier.predict(dataset)
        preds.index.name = "object_id"

        if not preds.index.equals(dataset.metadata.index):
            preds = preds.reindex(dataset.metadata.index)

        if preds.isnull().sum().sum() != 0:
            raise RuntimeError(f"Predictions contain NaNs for chunk {chunk}")

        # Save per-chunk predictions
        out_path = os.path.join(args.out_dir, f"predictions_chunk_{chunk}.h5")
        preds.to_hdf(out_path, key="predictions", mode="w")
        print(f"  Saved chunk predictions to {out_path}")
        print(f"  Chunk objects: {len(preds):,}")

        all_preds.append(preds)
        processed_chunks.append(chunk)

    if len(all_preds) == 0:
        raise RuntimeError("No predictions were generated.")

    combined = pd.concat(all_preds, axis=0)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()
    combined.index.name = "object_id"

    combined_path = os.path.join(args.out_dir, "predictions_combined.h5")
    combined.to_hdf(combined_path, key="predictions", mode="w")

    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"Processed chunks: {processed_chunks}")
    print(f"Combined prediction shape: {combined.shape}")
    print(f"Saved combined predictions to: {combined_path}")

    if args.score_flat:
        flat_score, n_scored = score_predictions(args.dataset, args.total_chunks, combined)
        print("\n" + "=" * 70)
        print("SCORING SUMMARY")
        print("=" * 70)
        print(f"Flat-weighted logloss: {flat_score:.5f}")
        print(f"Objects scored:        {n_scored:,}")
        print("Note: score is on overlapping labeled objects only, not necessarily the full test set.")

    return 0


if __name__ == "__main__":
    sys.exit(main())