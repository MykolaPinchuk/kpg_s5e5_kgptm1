#!/usr/bin/env python3
"""
Train XGBoost or LightGBM with optional domain features, target transform and exact CLI contract.

Saves:
 - model: models/<output_prefix>.pkl
 - preprocessor: models/<output_prefix>_preproc.pkl
 - metrics: models/<output_prefix>_metrics.json
 - val preds: models/<output_prefix>_val_preds.csv (id,Calories)
 - optional test preds: models/<output_prefix>_test_preds.csv
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

# optional lightgbm
try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None

# Top-level wrapper so joblib can pickle/unpickle xgb Booster objects
class BoosterWrapper:
    def __init__(self, booster):
        self.booster = booster

    def predict(self, X):
        dm = xgb.DMatrix(X)
        return self.booster.predict(dm)


def parse_args():
    p = argparse.ArgumentParser(description="Train XGB/LGB with engineered domain features")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=False)
    p.add_argument("--val_split", type=float, default=0.20)
    p.add_argument("--cv_folds", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_jobs", type=int, default=os.cpu_count())
    p.add_argument("--model", choices=["xgb", "lgb"], default="xgb")
    p.add_argument("--n_estimators", type=int, required=True)
    p.add_argument("--learning_rate", type=float, default=0.03)
    p.add_argument("--max_depth", type=int, default=7)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--colsample_bytree", type=float, default=1.0)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--min_child_weight", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--num_leaves", type=int, default=31)
    p.add_argument("--min_data_in_leaf", type=int, default=20)
    p.add_argument("--early_stopping", type=int, default=0)
    p.add_argument("--target_transform", choices=["none", "log1p"], default="log1p")
    p.add_argument("--use_domain_feats", choices=["0", "1"], default="0")
    p.add_argument("--output_dir", default="models")
    p.add_argument("--output_prefix", default="exp")
    p.add_argument("--save_val_preds", choices=["0", "1"], default="1")
    p.add_argument("--save_test_preds", choices=["0", "1"], default="0")
    return p.parse_args()


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def fit_preprocessor(df: pd.DataFrame, use_domain_feats: bool) -> Dict:
    """
    Fit preprocessor on training DataFrame only.
    """
    if "Calories" not in df.columns:
        raise ValueError("Expected target column 'Calories' not found in training file.")
    id_col = "id" if "id" in df.columns else None

    # numeric and categorical columns (exclude target and id)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(["Calories", "id"], errors="ignore").tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # numeric medians from TRAIN only
    numeric_medians = df[numeric_cols].median()

    # categorical mappings: label encoding for <=100 unique, else frequency mapping
    cat_mappings = {}
    for col in cat_cols:
        nunique = df[col].nunique(dropna=False)
        if nunique <= 100:
            _, uniques = pd.factorize(df[col], sort=False)
            mapping = {val: int(i) for i, val in enumerate(uniques)}
            cat_mappings[col] = {"type": "label", "mapping": mapping}
        else:
            freqs = df[col].value_counts(normalize=True).to_dict()
            cat_mappings[col] = {"type": "freq", "mapping": freqs}

    # missing indicators for numeric cols with >=1% missing
    missing_ind_cols = []
    n_rows = len(df)
    for col in numeric_cols:
        miss_frac = df[col].isna().sum() / max(1, n_rows)
        if miss_frac >= 0.01:
            missing_ind_cols.append(f"{col}_missing_ind")

    # aggregates
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
    numeric_std = df[numeric_cols].std(numeric_only=True)
    const_cols = numeric_std[numeric_std == 0].index.tolist()
    numeric_cols = [c for c in numeric_cols if c not in const_cols]

    # domain features placeholders (gender dummies discovered during transform on train)
    domain_feature_names = []
    if use_domain_feats:
        domain_feature_names = [
            "Height_m",
            "BMI",
            "Duration_hours",
            "Age_sq",
            "BMI_sq",
            "HR_ratio",
            "HR_per_kg",
            "HR_per_BMI",
            "HR_x_Duration",
            "BMI_x_Duration",
            "Duration_bin",
            # Gender_* dummies inserted after fit discovery
            "EstCal_total",
            "log1p_Duration",
            "log1p_Heart_Rate",
            "log1p_BMI",
            "log1p_EstCal_total",
        ]

    feature_columns: List[str] = []
    feature_columns.extend(numeric_cols)
    feature_columns.extend(missing_ind_cols)
    feature_columns.extend(cat_cols)
    feature_columns.extend(domain_feature_names)
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
        "use_domain_feats": bool(use_domain_feats),
    }
    return preproc


def _compute_domain_feats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain features using exact formulas from DS plan.
    Returns a copy with new columns appended.
    """
    df2 = df.copy()

    # Height_m = Height/100.0 if Height>3 else Height
    if "Height" in df2.columns:
        df2["Height_m"] = df2["Height"].apply(lambda h: (h / 100.0) if pd.notna(h) and h > 3 else h)
    else:
        df2["Height_m"] = np.nan

    # BMI = Weight / (Height_m**2 + 1e-9)
    def _bmi(row):
        wt = row.get("Weight", np.nan)
        hm = row.get("Height_m", np.nan)
        if pd.isna(wt) or pd.isna(hm):
            return np.nan
        return wt / (hm ** 2 + 1e-9)
    df2["BMI"] = df2.apply(_bmi, axis=1)

    # Duration_hours = Duration/60.0
    df2["Duration_hours"] = df2["Duration"] / 60.0 if "Duration" in df2.columns else np.nan

    # Age_sq = Age**2
    df2["Age_sq"] = df2["Age"] ** 2 if "Age" in df2.columns else np.nan

    # BMI_sq = BMI**2
    df2["BMI_sq"] = df2["BMI"] ** 2

    # HR_ratio = Heart_Rate / max(1.0, 220.0 - Age)
    def _hr_ratio(row):
        hr = row.get("Heart_Rate", np.nan)
        age = row.get("Age", 0.0)
        if pd.isna(hr):
            return np.nan
        denom = max(1.0, 220.0 - (age if not pd.isna(age) else 0.0))
        return hr / denom
    df2["HR_ratio"] = df2.apply(_hr_ratio, axis=1)

    # HR_per_kg = Heart_Rate / (Weight + 1e-6)
    df2["HR_per_kg"] = df2.apply(lambda r: (r["Heart_Rate"] / (r["Weight"] + 1e-6)) if pd.notna(r.get("Heart_Rate", np.nan)) and pd.notna(r.get("Weight", np.nan)) else np.nan, axis=1)

    # HR_per_BMI = Heart_Rate / (BMI + 1e-6)
    df2["HR_per_BMI"] = df2.apply(lambda r: (r["Heart_Rate"] / (r["BMI"] + 1e-6)) if pd.notna(r.get("Heart_Rate", np.nan)) and pd.notna(r.get("BMI", np.nan)) else np.nan, axis=1)

    # HR_x_Duration = Heart_Rate * Duration
    df2["HR_x_Duration"] = df2.apply(lambda r: (r.get("Heart_Rate", np.nan) * r.get("Duration", np.nan)) if pd.notna(r.get("Heart_Rate", np.nan)) and pd.notna(r.get("Duration", np.nan)) else np.nan, axis=1)

    # BMI_x_Duration = BMI * Duration
    df2["BMI_x_Duration"] = df2.apply(lambda r: (r.get("BMI", np.nan) * r.get("Duration", np.nan)) if pd.notna(r.get("BMI", np.nan)) and pd.notna(r.get("Duration", np.nan)) else np.nan, axis=1)

    # Duration_bin = np.digitize(Duration, bins=[10,20,30,45,60,90,120])
    if "Duration" in df2.columns:
        df2["Duration_bin"] = np.digitize(df2["Duration"].fillna(0).values, bins=[10, 20, 30, 45, 60, 90, 120])
    else:
        df2["Duration_bin"] = 0

    # Gender one-hot / dummy_na
    if "Gender" in df2.columns:
        dummies = pd.get_dummies(df2["Gender"].astype(object), prefix="Gender", dummy_na=True)
        # deterministic order
        for c in sorted(dummies.columns):
            df2[c] = dummies[c]
    else:
        df2["Gender_nan"] = 1.0

    # EstCal_total per DS formulas (male_per_min, female_per_min), clamp negative to 0, log1p(EstCal_total)
    def _est_cal(row):
        hr = row.get("Heart_Rate", np.nan)
        wt = row.get("Weight", np.nan)
        age = row.get("Age", np.nan)
        dur = row.get("Duration", 0.0)
        if pd.isna(hr) or pd.isna(wt) or pd.isna(age):
            return 0.0
        male_per_min = (-55.0969 + (0.6309 * hr) + (0.1988 * wt) + (0.2017 * age)) / 4.184
        female_per_min = (-20.4022 + (0.4472 * hr) - (0.1263 * wt) + (0.074 * age)) / 4.184
        g = row.get("Gender", None)
        per_min = female_per_min
        if isinstance(g, str):
            if g.lower().startswith("m"):
                per_min = male_per_min
            elif g.lower().startswith("f"):
                per_min = female_per_min
        else:
            per_min = 0.5 * (male_per_min + female_per_min)
        total = per_min * max(0.0, float(dur if not pd.isna(dur) else 0.0))
        return float(max(0.0, total))
    df2["EstCal_total"] = df2.apply(_est_cal, axis=1)

    # Log transforms: log1p(Duration), log1p(Heart_Rate), log1p(BMI), log1p(EstCal_total)
    df2["log1p_Duration"] = np.log1p(df2["Duration"].fillna(0.0))
    df2["log1p_Heart_Rate"] = np.log1p(df2["Heart_Rate"].fillna(0.0))
    df2["log1p_BMI"] = np.log1p(df2["BMI"].fillna(0.0))
    df2["log1p_EstCal_total"] = np.log1p(df2["EstCal_total"].fillna(0.0))

    return df2


def apply_preprocessor(df: pd.DataFrame, preproc: Dict, is_train: bool = False) -> pd.DataFrame:
    """
    Apply preprocessor dict to df and return X ordered by preproc['feature_columns'].
    """
    numeric_cols = preproc["numeric_cols"]
    cat_cols = preproc["cat_cols"]
    numeric_medians = pd.Series(preproc["numeric_medians"])
    missing_ind_cols = preproc["missing_ind_cols"]
    agg_feature_names = preproc["agg_feature_names"]
    feature_columns = preproc["feature_columns"]
    use_domain_feats = preproc.get("use_domain_feats", False)

    df_proc = df.copy()

    # Compute domain features first so they are available for imputation/mapping
    if use_domain_feats:
        df_proc = _compute_domain_feats(df_proc)

    # preserve ids separately
    id_col = preproc.get("id_col")
    ids = df_proc[id_col] if id_col and id_col in df_proc.columns else pd.Series(range(len(df_proc)), name="id")

    # numeric present (only those in numeric_cols that exist in df)
    numeric_present = [c for c in numeric_cols if c in df_proc.columns]
    numeric_df = df_proc[numeric_present].copy() if numeric_present else pd.DataFrame(index=df_proc.index)

    # aggregates before imputation
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

    # Impute numeric present with medians from TRAIN only
    for col in numeric_present:
        median = numeric_medians.get(col, 0.0)
        df_proc[col] = df_proc[col].fillna(median)

    # Map categoricals using saved mappings
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
        if col == id_col:
            # do not include id in features
            continue
        if col in df_proc.columns:
            features[col] = df_proc[col].astype(np.float32)
        else:
            features[col] = pd.Series(np.zeros(len(df_proc)), index=df_proc.index, dtype=np.float32)

    X = pd.DataFrame(features, index=df_proc.index)[[c for c in feature_columns if c != id_col]]
    return X, ids


def rmsle_from_original(y_true, y_pred):
    y_pred = np.clip(y_pred, 0.0, None)
    eps = 1e-9
    return float(np.sqrt(mean_squared_log_error(y_true + eps, y_pred + eps)))


def main():
    args = parse_args()
    start_time = time.time()

    _ensure_dir(args.output_dir)

    # paths
    model_path = os.path.join(args.output_dir, f"{args.output_prefix}.pkl")
    preproc_path = os.path.join(args.output_dir, f"{args.output_prefix}_preproc.pkl")
    metrics_path = os.path.join(args.output_dir, f"{args.output_prefix}_metrics.json")
    val_preds_path = os.path.join(args.output_dir, f"{args.output_prefix}_val_preds.csv")
    test_preds_path = os.path.join(args.output_dir, f"{args.output_prefix}_test_preds.csv")

    # load train
    if not os.path.exists(args.train_csv):
        print(f"Training file not found: {args.train_csv}", flush=True)
        return 2
    df = pd.read_csv(args.train_csv)
    if "Calories" not in df.columns:
        print("Expected target column 'Calories' not found in training file.", flush=True)
        return 3

    # prepare train/val
    if args.val_csv:
        if not os.path.exists(args.val_csv):
            print(f"Val file not found: {args.val_csv}", flush=True)
            return 4
        val_df = pd.read_csv(args.val_csv)
        train_df = df
    else:
        train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=args.seed)

    # Fit preprocessor on TRAIN only
    use_domain = args.use_domain_feats == "1"
    preproc = fit_preprocessor(train_df, use_domain_feats=use_domain)

    # If domain features enabled, discover any gender dummies from train and ensure they are in feature_columns deterministically
    if preproc.get("use_domain_feats"):
        train_with_dom = _compute_domain_feats(train_df)
        gender_cols = sorted([c for c in train_with_dom.columns if c.startswith("Gender_")])
        # insert gender cols after cat columns if not already present
        fc = [c for c in preproc["feature_columns"] if not c.startswith("Gender_")]
        insert_pos = len(preproc["numeric_cols"]) + len(preproc["missing_ind_cols"]) + len(preproc["cat_cols"])
        # insert preserving order
        for i, g in enumerate(gender_cols):
            fc.insert(insert_pos + i, g)
        preproc["feature_columns"] = fc

    # Save preprocessor now (will include feature_columns and mappings)
    joblib.dump(preproc, preproc_path)

    # Apply preprocessor to train and val
    X_train, _ = apply_preprocessor(train_df, preproc, is_train=True)
    X_val, val_ids = apply_preprocessor(val_df, preproc, is_train=False)

    # Target transform
    if args.target_transform == "log1p":
        y_train = np.log1p(train_df["Calories"].values.astype(np.float64))
        y_val = np.log1p(val_df["Calories"].values.astype(np.float64))
        inverse_fn = np.expm1
    else:
        y_train = train_df["Calories"].values.astype(np.float64)
        y_val = val_df["Calories"].values.astype(np.float64)
        inverse_fn = lambda x: x

    X_train_arr = X_train.values.astype(np.float32)
    X_val_arr = X_val.values.astype(np.float32)

    # Train model
    model_obj = None
    best_iteration = None
    if args.model == "xgb":
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
            "lambda": float(args.reg_lambda),
            "gamma": float(args.gamma),
            "verbosity": 0,
            "nthread": int(args.n_jobs),
        }
        num_round = int(args.n_estimators)
        evals = [(dval, "validation")]
        if args.early_stopping and args.early_stopping > 0:
            booster = xgb.train(params, dtrain, num_boost_round=num_round, evals=evals, early_stopping_rounds=int(args.early_stopping), verbose_eval=False)
        else:
            booster = xgb.train(params, dtrain, num_boost_round=num_round, evals=evals, verbose_eval=False)
        model_obj = BoosterWrapper(booster)
        try:
            best_iteration = int(getattr(booster, "best_iteration", None) or getattr(booster, "best_ntree_limit", None) or num_round)
        except Exception:
            best_iteration = num_round
    else:
        # LightGBM path
        if lgb is None:
            raise RuntimeError("LightGBM requested but not installed.")
        lgb_params = {
            "objective": "regression",
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "min_data_in_leaf": int(args.min_data_in_leaf),
            "max_depth": int(args.max_depth),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "reg_lambda": float(args.reg_lambda),
            "min_child_weight": float(args.min_child_weight),
            "n_jobs": int(args.n_jobs),
            "verbosity": -1,
        }
        model_lgb = lgb.LGBMRegressor(n_estimators=int(args.n_estimators), **lgb_params)
        if args.early_stopping and args.early_stopping > 0:
            model_lgb.fit(X_train_arr, y_train, eval_set=[(X_val_arr, y_val)], early_stopping_rounds=int(args.early_stopping), verbose=False)
            best_iteration = getattr(model_lgb, "best_iteration_", int(args.n_estimators))
        else:
            model_lgb.fit(X_train_arr, y_train, verbose=False)
            best_iteration = int(args.n_estimators)
        model_obj = model_lgb

    # Predictions on validation (model outputs in transformed space)
    val_pred_trans = model_obj.predict(X_val_arr)
    val_pred = inverse_fn(val_pred_trans)
    val_pred = np.clip(val_pred, 0.0, None)

    # Metrics
    if args.target_transform == "log1p":
        # y_val is in log-space
        val_rmse_log = float(np.sqrt(mean_squared_error(y_val, val_pred_trans)))
        val_rmsle = rmsle_from_original(np.expm1(y_val), val_pred)
    else:
        val_rmse_log = None
        val_rmsle = rmsle_from_original(y_val, val_pred)

    elapsed = time.time() - start_time

    # Save artifacts
    joblib.dump(model_obj, model_path)
    joblib.dump(preproc, preproc_path)

    # Save val preds if requested
    if args.save_val_preds == "1":
        out_val = pd.DataFrame({"id": val_ids, "Calories": val_pred})
        out_val.to_csv(val_preds_path, index=False)

    # Optionally save test preds using data/test.csv if requested
    test_saved = False
    if args.save_test_preds == "1":
        test_file = os.path.join("data", "test.csv")
        if os.path.exists(test_file):
            df_test = pd.read_csv(test_file)
            X_test, test_ids = apply_preprocessor(df_test, preproc, is_train=False)
            X_test_arr = X_test.values.astype(np.float32)
            test_pred_trans = model_obj.predict(X_test_arr)
            test_pred = inverse_fn(test_pred_trans)
            test_pred = np.clip(test_pred, 0.0, None)
            out_test = pd.DataFrame({"id": test_ids, "Calories": test_pred})
            out_test.to_csv(test_preds_path, index=False)
            test_saved = True

    # Prepare metrics dict
    hyperparams = {
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "max_depth": int(args.max_depth),
        "min_child_weight": float(args.min_child_weight),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "reg_lambda": float(args.reg_lambda),
        "gamma": float(args.gamma),
    }
    metrics = {
        "val_rmsle": val_rmsle,
        "val_rmse_log": val_rmse_log,
        "n_train": int(X_train_arr.shape[0]),
        "n_val": int(X_val_arr.shape[0]),
        "best_iteration": int(best_iteration) if best_iteration is not None else None,
        "hyperparams": hyperparams,
        "n_features": int(X_train_arr.shape[1]),
        "feature_columns": preproc["feature_columns"],
        "preprocessor_path": preproc_path,
        "model_path": model_path,
        "elapsed_seconds": float(elapsed),
        "random_state": int(args.seed),
    }

    # Write metrics JSON
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved preprocessor to {preproc_path}")
    print(f"Saved metrics to {metrics_path}")
    if args.save_val_preds == "1":
        print(f"Saved validation predictions to {val_preds_path}")
    if test_saved:
        print(f"Saved test predictions to {test_preds_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("ERROR:", e, flush=True)
        raise