#!/usr/bin/env python3
"""
Train XGBoost on log1p(target) with feature engineering & save artifacts + metrics.

See task for exact CLI and preprocessing contract.
"""
import argparse
import json
import os
import time
import joblib
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import xgboost as xgb
from xgboost import XGBRegressor

# Top-level wrapper so joblib can pickle/unpickle it (must be importable by module path)
class BoosterWrapper:
    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        dm = xgb.DMatrix(X)
        return self.booster.predict(dm)


def parse_args():
    p = argparse.ArgumentParser(description="Train XGB on log1p(target) with engineered features")
    p.add_argument("--train", required=True)
    p.add_argument("--model_out", required=True)
    p.add_argument("--metrics_out", required=True)
    p.add_argument("--preproc_out", required=True)
    p.add_argument("--val_size", type=float, default=0.20)
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
    p.add_argument("--early_stopping_rounds", type=int, required=True)
    p.add_argument("--n_jobs", type=int, required=True)
    return p.parse_args()


def _ensure_path(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def fit_preprocessor(df: pd.DataFrame, random_state: int = 42):
    # Expect target "Calories"
    if "Calories" not in df.columns:
        raise ValueError("Expected target column 'Calories' not found in training file.")
    # Preserve id if present
    id_col = "id" if "id" in df.columns else None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(["Calories", "id"], errors="ignore").tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    train_numeric = df[numeric_cols].copy()

    # numeric medians
    numeric_medians = train_numeric.median()

    # cat mappings
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

    # missing indicators
    missing_ind_cols = []
    n_rows = len(df)
    for col in numeric_cols:
        miss_frac = df[col].isna().sum() / max(1, n_rows)
        if miss_frac >= 0.01:
            missing_ind_cols.append(f"{col}_missing_ind")

    # aggregates computed on numeric cols (before imputation)
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

    # drop constant numeric cols (std == 0)
    numeric_std = train_numeric.std(numeric_only=True)
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


def apply_preprocessor(df: pd.DataFrame, preproc: Dict, is_train: bool = False) -> pd.DataFrame:
    """
    Apply preprocessor to df and return DataFrame with columns ordered as preproc['feature_columns'].
    """
    numeric_cols = preproc["numeric_cols"]
    cat_cols = preproc["cat_cols"]
    numeric_medians = pd.Series(preproc["numeric_medians"])
    missing_ind_cols = preproc["missing_ind_cols"]
    agg_feature_names = preproc["agg_feature_names"]

    # Work on a copy
    df_proc = df.copy()

    # Compute aggregates using numeric columns present in df (may miss some if constant dropped)
    # Count missing per row before impute (among numeric columns considered)
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

    # Missing indicators (based on original values)
    for ind in missing_ind_cols:
        # ind is like "{col}_missing_ind"
        col = ind.rsplit("_missing_ind", 1)[0]
        df_proc[ind] = df_proc[col].isna().astype(int) if col in df_proc.columns else 0

    # Impute numeric columns with medians (use medians for columns that exist)
    for col in numeric_present:
        median = numeric_medians.get(col, 0.0)
        df_proc[col] = df_proc[col].fillna(median)

    # Map categorical columns
    for col in cat_cols:
        if col not in df_proc.columns:
            # create zeros if absent
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
    for col in preproc["feature_columns"]:
        if col in df_proc.columns:
            features[col] = df_proc[col].astype(np.float32)
        else:
            # missing column (e.g., numeric dropped earlier) fill 0
            features[col] = pd.Series(np.zeros(len(df_proc)), index=df_proc.index, dtype=np.float32)

    X = pd.DataFrame(features, index=df_proc.index)[preproc["feature_columns"]]

    return X


def rmsle_from_original(y_true, y_pred):
    y_pred = np.clip(y_pred, 0.0, None)
    eps = 1e-9
    return float(np.sqrt(mean_squared_log_error(y_true + eps, y_pred + eps)))


def main():
    args = parse_args()

    start_time = time.time()

    if not os.path.exists(args.train):
        print(f"Training file not found: {args.train}", flush=True)
        return 2

    # Load
    df = pd.read_csv(args.train)
    if "Calories" not in df.columns:
        print("Expected target column 'Calories' not found in training file.", flush=True)
        return 3

    # Split
    train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=args.random_state)

    # Fit preprocessor on train
    preproc = fit_preprocessor(train_df, random_state=args.random_state)

    # Apply to train and val
    X_train = apply_preprocessor(train_df, preproc, is_train=True)
    X_val = apply_preprocessor(val_df, preproc, is_train=False)

    # Prepare y (log1p)
    y_train = np.log1p(train_df["Calories"].values.astype(np.float64))
    y_val = np.log1p(val_df["Calories"].values.astype(np.float64))

    # Convert X to float32 arrays
    X_train_arr = X_train.values.astype(np.float32)
    X_val_arr = X_val.values.astype(np.float32)

    # Use xgboost.train (Booster) for compatibility across versions
    dtrain = xgb.DMatrix(X_train_arr, label=y_train)
    dval = xgb.DMatrix(X_val_arr, label=y_val)
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eta": float(args.learning_rate),
        "max_depth": int(args.max_depth),
        "min_child_weight": float(args.min_child_weight),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "alpha": float(args.reg_alpha),
        "lambda": float(args.reg_lambda),
        "gamma": float(args.gamma),
        "verbosity": 0,
        "nthread": int(args.n_jobs),
    }
    num_round = int(args.n_estimators)
    early_stopping = int(getattr(args, "early_stopping_rounds", 0)) if getattr(args, "early_stopping_rounds", None) is not None else None

    evals = [(dval, "validation")]
    if early_stopping and early_stopping > 0:
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_round,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )
    else:
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_round,
            evals=evals,
            verbose_eval=False,
        )

    # Wrap Booster with a simple predict interface compatible with joblib.load usage in other scripts
    model = BoosterWrapper(booster)

    # Best iteration
    best_iter = None
    try:
        best_iter = int(getattr(booster, "best_iteration", None) or getattr(booster, "best_ntree_limit", None) or num_round)
    except Exception:
        best_iter = num_round

    # Predictions: model outputs log1p predictions
    val_pred_log = model.predict(X_val_arr)
    val_pred = np.expm1(val_pred_log)
    val_pred = np.clip(val_pred, 0.0, None)

    # Metrics
    # val_rmse_log: RMSE on log-space between y_val (log1p) and val_pred_log
    val_rmse_log = float(np.sqrt(mean_squared_error(y_val, val_pred_log)))
    val_rmsle = rmsle_from_original(np.expm1(y_val), val_pred)

    elapsed = time.time() - start_time

    # Save artifacts
    _ensure_path(args.model_out)
    _ensure_path(args.preproc_out)
    _ensure_path(args.metrics_out)

    joblib.dump(model, args.model_out)
    joblib.dump(preproc, args.preproc_out)

    metrics = {
        "val_rmsle": val_rmsle,
        "val_rmse_log": val_rmse_log,
        "n_train": int(len(X_train_arr)),
        "n_val": int(len(X_val_arr)),
        "best_iteration": int(best_iter) if best_iter is not None else None,
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
        "elapsed_seconds": float(elapsed),
        "random_state": int(args.random_state),
    }

    # Write metrics JSON
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {args.model_out}")
    print(f"Saved preprocessor to {args.preproc_out}")
    print(f"Saved metrics to {args.metrics_out}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("ERROR:", e, flush=True)
        raise