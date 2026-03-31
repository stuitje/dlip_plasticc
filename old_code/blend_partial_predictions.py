#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import avocado


def score_flat(dataset_name, pred_df):
    dataset = avocado.load(dataset_name, metadata_only=True)
    metadata = dataset.metadata

    common_idx = metadata.index.intersection(pred_df.index)
    true_classes = metadata.loc[common_idx, "class"]
    pred_aligned = pred_df.loc[common_idx].copy()

    try:
        pred_aligned.columns = pred_aligned.columns.astype(true_classes.dtype)
    except Exception:
        pass

    # ignore anomaly classes not predicted by model
    known_mask = true_classes.isin(set(pred_aligned.columns))
    true_classes = true_classes.loc[known_mask]
    pred_aligned = pred_aligned.loc[known_mask]

    score = avocado.weighted_multi_logloss(
        true_classes,
        pred_aligned,
        class_weights=avocado.plasticc.plasticc_flat_weights,
    )
    return score, len(true_classes)


def normalize_rows(df):
    return df.div(df.sum(axis=1), axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-a", required=True, help="First prediction HDF")
    parser.add_argument("--pred-b", required=True, help="Second prediction HDF")
    parser.add_argument("--dataset", default="plasticc_test")
    parser.add_argument("--out", required=True, help="Output blended HDF")
    parser.add_argument("--weight-a", type=float, default=None,
                        help="Fixed weight for pred-a. If omitted, do a grid search.")
    parser.add_argument("--grid-step", type=float, default=0.05)
    args = parser.parse_args()

    pred_a = pd.read_hdf(args.pred_a, key="predictions")
    pred_b = pd.read_hdf(args.pred_b, key="predictions")

    # align rows
    common_idx = pred_a.index.intersection(pred_b.index)
    pred_a = pred_a.loc[common_idx].sort_index()
    pred_b = pred_b.loc[common_idx].sort_index()

    # align columns
    common_cols = pred_a.columns.intersection(pred_b.columns)
    pred_a = pred_a[common_cols]
    pred_b = pred_b[common_cols]

    # normalize just in case
    pred_a = normalize_rows(pred_a)
    pred_b = normalize_rows(pred_b)

    if args.weight_a is None:
        print("Searching blend weights...")
        best_score = np.inf
        best_w = None
        best_blend = None

        weights = np.arange(0.0, 1.0 + 1e-9, args.grid_step)
        for w in weights:
            blended = w * pred_a + (1.0 - w) * pred_b
            blended = normalize_rows(blended)
            score, n_scored = score_flat(args.dataset, blended)
            print(f"weight_a={w:.2f}  flat_logloss={score:.5f}  n={n_scored:,}")

            if score < best_score:
                best_score = score
                best_w = w
                best_blend = blended

        print("\nBest blend:")
        print(f"  weight_a = {best_w:.2f}")
        print(f"  flat-weighted logloss = {best_score:.5f}")

        blended = best_blend
    else:
        w = args.weight_a
        blended = w * pred_a + (1.0 - w) * pred_b
        blended = normalize_rows(blended)
        score, n_scored = score_flat(args.dataset, blended)
        print(f"Fixed blend weight_a={w:.2f}")
        print(f"Flat-weighted logloss = {score:.5f}  on {n_scored:,} objects")

    blended.index.name = "object_id"
    blended.to_hdf(args.out, key="predictions", mode="w")
    print(f"Saved blended predictions to: {args.out}")


if __name__ == "__main__":
    main()