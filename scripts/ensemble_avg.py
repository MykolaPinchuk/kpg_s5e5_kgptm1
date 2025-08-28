#!/usr/bin/env python3
"""
Ensemble average predictions from multiple models + preprocessors.

CLI:
  --models (one or more)
  --preproc (one or more or single)
  --test
  --out
  --metrics_out
  --id_col (default 'id')
  --n_jobs
"""
import argparse
import json
import os
import time
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# Provide BoosterWrapper at top-level so joblib can unpickle objects saved that reference this class
class BoosterWrapper:
    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        dm = xgb.DMatrix(X)
        return self.booster.predict(dm)


def parse_args():
    p = argparse.ArgumentParser(description="Ensemble average predictions from models")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--preproc", nargs="+", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--metrics_out", required=True)
    p.add_argument("--id_col", default="id")
    p.add_argument("--n_jobs", type=int, default=1)
    return p.parse_args()


def apply_preprocessor(df: pd.DataFrame, preproc: Dict) -> pd.DataFrame:
    """
    Apply preprocessor dict (as saved by train_xgb_log_feats) to df and return X (ordered).
    """
    numeric_cols = preproc["numeric_cols"]
    cat_cols = preproc["cat_cols"]
    numeric_medians = pd.Series(preproc["numeric_medians"])
    missing_ind_cols = preproc["missing_ind_cols"]
    agg_feature_names = preproc["agg_feature_names"]
    feature_columns = preproc["feature_columns"]

    df_proc = df.copy()

    # numeric present (only those in numeric_cols that exist in df)
    numeric_present = [c for c in numeric_cols if c in df_proc.columns]
    numeric_df = df_proc[numeric_present].copy() if numeric_present else pd.DataFrame(index=df_proc.index)

    row_num_missing = numeric_df.isna().sum(axis=1)
    numeric_count = numeric_df.count(axis=1)
    numeric_sum = numeric_df.sum(axis=1)
    numeric_mean = numeric_df.mean(axis=1)
    numeric_std = numeric_df.std(axis=1)
    numeric_min = numeric_df.min(axis=1)
    numeric_max = numeric_df.max(axis=1)
    numeric_range = numeric_max - numeric_min

    # Missing indicators
    for ind in missing_ind_cols:
        col = ind.rsplit("_missing_ind", 1)[0]
        if col in df_proc.columns:
            df_proc[ind] = df_proc[col].isna().astype(int)
        else:
            df_proc[ind] = 0

    # Impute numeric present with medians
    for col in numeric_present:
        median = numeric_medians.get(col, 0.0)
        df_proc[col] = df_proc[col].fillna(median)

    # Map categoricals
    for col in cat_cols:
        if col not in df_proc.columns:
            df_proc[col] = 0
            continue
        mapinfo = preproc["cat_mappings"].get(col)
        if not mapinfo:
            df_proc[col] = df_proc[col].astype("category").cat.codes
            continue
        if mapinfo["type"] == "label":
            mapping = mapinfo["mapping"]
            df_proc[col] = df_proc[col].map(mapping).fillna(-1).astype(int)
        else:
            mapping = mapinfo["mapping"]
            df_proc[col] = df_proc[col].map(mapping).fillna(0.0).astype(float)

    # Attach aggregates
    agg_vals = {
        "numeric_count": numeric_count,
        "numeric_sum": numeric_sum,
        "numeric_mean": numeric_mean,
        "numeric_std": numeric_std,
        "numeric_min": numeric_min,
        "numeric_max": numeric_max,
        "numeric_range": numeric_range,
        "row_num_missing": row_num_missing,
    }
    for name in agg_feature_names:
        df_proc[name] = agg_vals.get(name, 0.0)

    # Build final DataFrame with feature_columns (missing ones filled with 0)
    features = {}
    for col in feature_columns:
        if col in df_proc.columns:
            features[col] = df_proc[col].astype(np.float32)
        else:
            features[col] = pd.Series(np.zeros(len(df_proc)), index=df_proc.index, dtype=np.float32)

    X = pd.DataFrame(features, index=df_proc.index)[feature_columns]
    return X


def _ensure_path(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def main():
    args = parse_args()
    start = time.time()

    for pth in args.models:
        if not os.path.exists(pth):
            print(f"Model file not found: {pth}", flush=True)
            return 2
    for pth in args.preproc:
        if not os.path.exists(pth):
            print(f"Preproc file not found: {pth}", flush=True)
            return 3
    if not os.path.exists(args.test):
        print(f"Test file not found: {args.test}", flush=True)
        return 4

    _ensure_path(args.out)
    _ensure_path(args.metrics_out)

    # Load test
    df_test = pd.read_csv(args.test)
    ids = df_test[args.id_col] if args.id_col in df_test.columns else pd.Series(range(len(df_test)), name=args.id_col)

    preds_list = []
    n_models = len(args.models)

    # If single preproc supplied but multiple models, reuse it
    preproc_paths = args.preproc
    if len(preproc_paths) == 1 and n_models > 1:
        preproc_paths = preproc_paths * n_models
    if len(preproc_paths) != n_models:
        # allow the user to pass equal counts; otherwise error
        print("Number of preproc paths must be 1 or equal to number of models", flush=True)
        return 5

    for model_path, preproc_path in zip(args.models, preproc_paths):
        preproc = joblib.load(preproc_path)
        X = apply_preprocessor(df_test, preproc)
        X_arr = X.values.astype(np.float32)

        model = joblib.load(model_path)
        preds_log = model.predict(X_arr)
        preds = np.expm1(preds_log)
        preds = np.clip(preds, 0.0, None)
        preds_list.append(preds)

    # Stack and average
    preds_stack = np.vstack(preds_list)
    final_preds = np.mean(preds_stack, axis=0)

    out_df = pd.DataFrame({args.id_col: ids, "Calories": final_preds})
    out_df.to_csv(args.out, index=False)

    elapsed = time.time() - start
    metrics = {
        "ensemble_models": args.models,
        "n_models": int(n_models),
        "n_rows": int(len(out_df)),
        "out_path": args.out,
        "elapsed_seconds": float(elapsed),
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote ensemble predictions to {args.out}")
    print(f"Wrote ensemble metrics to {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())