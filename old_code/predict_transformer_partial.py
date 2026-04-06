#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm
import avocado

from avocado.sequence_featurizer import PlasticcSequenceFeaturizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--dataset", default="plasticc_test")
    parser.add_argument("--total-chunks", type=int, default=500)
    parser.add_argument("--chunks", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=350)
    parser.add_argument("--score-flat", action="store_true")
    return parser.parse_args()


def score_predictions(dataset_name, pred_df):
    dataset_full = avocado.load(dataset_name, metadata_only=True)
    metadata = dataset_full.metadata

    common_idx = metadata.index.intersection(pred_df.index)
    true_classes = metadata.loc[common_idx, "class"]
    pred_aligned = pred_df.loc[common_idx].copy()

    try:
        pred_aligned.columns = pred_aligned.columns.astype(true_classes.dtype)
    except Exception:
        pass

    known_mask = true_classes.isin(set(pred_aligned.columns))
    true_classes = true_classes.loc[known_mask]
    pred_aligned = pred_aligned.loc[known_mask]

    score = avocado.weighted_multi_logloss(
        true_classes,
        pred_aligned,
        class_weights=avocado.plasticc.plasticc_flat_weights,
    )
    return score, len(true_classes)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading classifier '{args.classifier}'...")
    classifier = avocado.Classifier.load(args.classifier)

    featurizer = PlasticcSequenceFeaturizer(seq_len=args.seq_len)
    all_preds = []

    for chunk in tqdm(args.chunks, desc="Chunks", dynamic_ncols=True):
        print(f"\n=== Processing chunk {chunk}/{args.total_chunks - 1} ===")

        dataset = avocado.load(
            args.dataset,
            chunk=chunk,
            num_chunks=args.total_chunks,
            metadata_only=False,
        )

        dataset.extract_raw_features(featurizer)
        preds = classifier.predict(dataset)
        preds.index.name = "object_id"

        out_path = os.path.join(args.out_dir, f"predictions_chunk_{chunk}.h5")
        preds.to_hdf(out_path, key="predictions", mode="w")
        print(f"  Saved {len(preds):,} predictions to {out_path}")

        all_preds.append(preds)

    combined = pd.concat(all_preds).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    combined.index.name = "object_id"

    combined_path = os.path.join(args.out_dir, "predictions_combined.h5")
    combined.to_hdf(combined_path, key="predictions", mode="w")

    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"Combined prediction shape: {combined.shape}")
    print(f"Saved combined predictions to: {combined_path}")

    if args.score_flat:
        score, n_scored = score_predictions(args.dataset, combined)
        print("\n" + "=" * 70)
        print("SCORING SUMMARY")
        print("=" * 70)
        print(f"Flat-weighted logloss: {score:.5f}")
        print(f"Objects scored:        {n_scored:,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())