#!/usr/bin/env python3
"""
Retrain selected config on full train.csv and produce test predictions.

CLI arguments mirror train_xgb_log_feats with test/pred_out and without required early_stopping_rounds.
"""
import argparse
import json
import os
import time
import joblib
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from xgboost import XGBRegressor


def parse_args():
    p = argparse.ArgumentParser(description="Train final model on full train and produce test preds")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--model_out", required=True)
    p.add_argument("--preproc_out", required=True)
    p.add_argument("--metrics_out", required=True)
    p.add_argument("--pred_out", required=True)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--n_estimators", type=int, required=True)
    p.add_argument("--learning_rate", type=float, required=True)
    p.add_argument("--max_depth", type=int, required=True)
    p.add_argument("--min_child_weight", type=float, required=True)
    p.add_argument("--subsample", type=float, required=True)
    p.add_argument("--colsample_bytree", type=float, required=True)
    p.add_argument("--reg_alpha", type=float, required=True)
    p.add_argument("--reg_lambda", type=float, required=True)
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--n_jobs", type=int, default=1)
    return p.parse_args()


def _ensure_path(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def fit_preprocessor(df: pd.DataFrame) -> Dict:
    if "Calories" not in df.columns:
        raise ValueError("Expected target column 'Calories' not found in training file.")
    id_col = "id" if "id" in df.columns else None
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(["Calories", "id"], errors="ignore").tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_medians = df[numeric_cols].median()
    cat_mappings = {}
    for col in cat_cols:
        nunique = df[col].nunique(dropna=False)
        if nunique <= 100:
            codes, uniques = pd.factorize(df[col], sort=False)
            mapping = {val: int(i) for i, val in enumerate(uniques)}
            cat_mappings[col] = {"type": "label", "mapping": mapping}
        else:
            freqs = df[col].value_counts(normalize=True).to_dict()
            cat_mappings[col] = {"type": "freq", "mapping": freqs}

    missing_ind_cols = []
    n_rows = len(df)
    for col in numeric_cols:
        miss_frac = df[col].isna().sum() / max(1, n_rows)
        if miss_frac >= 0.01:
            missing_ind_cols.append(f"{col}_missing_ind")

    agg_feature_names = [
        "numeric_count",
        "numeric_sum",
        "numeric_mean",
        "numeric_std",
        "numeric_min",
        "numeric_max",
        "numeric_range",
        "row_num_missing",
    ]

    numeric_std = df[numeric_cols].std(numeric_only=True)
    const_cols = numeric_std[numeric_std == 0].index.tolist()
    numeric_cols = [c for c in numeric_cols if c not in const_cols]

    feature_columns = []
    feature_columns.extend(numeric_cols)
    feature_columns.extend(missing_ind_cols)
    feature_columns.extend(cat_cols)
    feature_columns.extend(agg_feature_names)

    preproc = {
        "numeric_cols": numeric_cols,
        "numeric_medians": numeric_medians.to_dict(),
        "cat_cols": cat_cols,
        "cat_mappings": cat_mappings,
        "missing_ind_cols": missing_ind_cols,
        "agg_feature_names": agg_feature_names,
        "feature_columns": feature_columns,
        "id_col": id_col,
    }
    return preproc


def apply_preprocessor(df: pd.DataFrame, preproc: Dict) -> pd.DataFrame:
    numeric_cols = preproc["numeric_cols"]
    cat_cols = preproc["cat_cols"]
    numeric_medians = pd.Series(preproc["numeric_medians"])
    missing_ind_cols = preproc["missing_ind_cols"]
    agg_feature_names = preproc["agg_feature_names"]
    feature_columns = preproc["feature_columns"]

    df_proc = df.copy()
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

    for ind in missing_ind_cols:
        col = ind.rsplit("_missing_ind", 1)[0]
        if col in df_proc.columns:
            df_proc[ind] = df_proc[col].isna().astype(int)
        else:
            df_proc[ind] = 0

    for col in numeric_present:
        median = numeric_medians.get(col, 0.0)
        df_proc[col] = df_proc[col].fillna(median)

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

    features = {}
    for col in feature_columns:
        if col in df_proc.columns:
            features[col] = df_proc[col].astype(np.float32)
        else:
            features[col] = pd.Series(np.zeros(len(df_proc)), index=df_proc.index, dtype=np.float32)

    X = pd.DataFrame(features, index=df_proc.index)[feature_columns]
    return X


def main():
    args = parse_args()
    start = time.time()

    if not os.path.exists(args.train):
        print(f"Training file not found: {args.train}", flush=True)
        return 2
    if not os.path.exists(args.test):
        print(f"Test file not found: {args.test}", flush=True)
        return 3

    df_train = pd.read_csv(args.train)
    if "Calories" not in df_train.columns:
        print("Expected target column 'Calories' not found in training file.", flush=True)
        return 4

    # Fit preprocessor on full train
    preproc = fit_preprocessor(df_train)

    X_train = apply_preprocessor(df_train, preproc)
    y_train = np.log1p(df_train["Calories"].values.astype(np.float64))
    X_train_arr = X_train.values.astype(np.float32)

    # Train model (no early stopping by default)
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        verbosity=0,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
    )

    model.fit(X_train_arr, y_train, verbose=False)

    best_iter = args.n_estimators

    # Save artifacts
    _ensure_path(args.model_out)
    _ensure_path(args.preproc_out)
    _ensure_path(args.metrics_out)
    _ensure_path(args.pred_out)

    joblib.dump(model, args.model_out)
    joblib.dump(preproc, args.preproc_out)

    # Produce test preds
    df_test = pd.read_csv(args.test)
    ids = df_test["id"] if "id" in df_test.columns else pd.Series(range(len(df_test)), name="id")
    X_test = apply_preprocessor(df_test, preproc)
    X_test_arr = X_test.values.astype(np.float32)

    preds_log = model.predict(X_test_arr)
    preds = np.expm1(preds_log)
    preds = np.clip(preds, 0.0, None)

    out_df = pd.DataFrame({"id": ids, "Calories": preds})
    out_df.to_csv(args.pred_out, index=False)

    elapsed = time.time() - start

    metrics = {
        "val_rmsle": None,
        "val_rmse_log": None,
        "n_train": int(len(X_train_arr)),
        "n_val": 0,
        "best_iteration": int(best_iter),
        "hyperparams": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_child_weight": args.min_child_weight,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "gamma": args.gamma,
        },
        "n_features": int(X_train_arr.shape[1]),
        "feature_columns": preproc["feature_columns"],
        "preprocessor_path": args.preproc_out,
        "model_path": args.model_out,
        "pred_out": args.pred_out,
        "elapsed_seconds": float(elapsed),
        "random_state": int(args.random_state),
    }

    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved final model to {args.model_out}")
    print(f"Saved preprocessor to {args.preproc_out}")
    print(f"Saved test predictions to {args.pred_out}")
    print(f"Saved metrics to {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())