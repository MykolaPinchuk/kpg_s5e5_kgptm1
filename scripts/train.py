#!/usr/bin/env python3
"""
Quick POC training script.

Usage:
    python3 scripts/train.py --train ./data/subsample_train.csv --model_out ./models/poc_model.pkl

This script:
- Loads a CSV with a "Calories" target column.
- Splits into train/validation.
- Trains an XGBoost regressor with simple defaults.
- Evaluates RMSLE on the validation set.
- Saves the trained model (joblib) and a small metrics JSON.
"""
import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor

def parse_args():
    p = argparse.ArgumentParser(description="Train a quick XGBoost POC model")
    p.add_argument("--train", "-t", default="./data/subsample_train.csv", help="Path to train CSV (must contain 'Calories' column)")
    p.add_argument("--model_out", "-m", default="./models/poc_model.pkl", help="Path to write model artifact (joblib)")
    p.add_argument("--val_size", type=float, default=0.2, help="Validation fraction")
    p.add_argument("--random_state", type=int, default=42, help="Random seed")
    p.add_argument("--n_estimators", type=int, default=100, help="XGBoost n_estimators")
    p.add_argument("--max_depth", type=int, default=6, help="XGBoost max_depth")
    return p.parse_args()

def rmsle(y_true, y_pred):
    # Ensure no negative predictions (clip) and avoid domain errors
    y_pred = np.clip(y_pred, 0, None)
    # Add small epsilon to avoid log(0)
    eps = 1e-9
    return float(np.sqrt(mean_squared_log_error(y_true + eps, y_pred + eps)))

def main():
    args = parse_args()

    if not os.path.exists(args.train):
        print(f"Training file not found: {args.train}", flush=True)
        return 2

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)

    print(f"Loading training data from {args.train} ...")
    df = pd.read_csv(args.train)

    if "Calories" not in df.columns:
        print("Expected target column 'Calories' not found in training file.", flush=True)
        return 3

    # Prepare features
    X = df.drop(columns=["Calories"])
    # Drop id if present from features, but keep if not present
    if "id" in X.columns:
        X = X.drop(columns=["id"])

    y = df["Calories"].values

    # Simple preprocessing: fill NA with median for numeric columns
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median())

    print(f"Feature matrix shape: {X.shape}; target shape: {y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.random_state
    )

    print("Training XGBoost regressor (fast defaults)...")
    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        verbosity=0,
        n_jobs=1,
    )

    model.fit(X_train, y_train)

    print("Predicting on validation set...")
    val_pred = model.predict(X_val)
    val_rmsle = rmsle(y_val, val_pred)
    print(f"Validation RMSLE: {val_rmsle:.6f}")

    # Save model and metrics
    joblib.dump(model, args.model_out)
    metrics = {
        "val_rmsle": val_rmsle,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "model_path": args.model_out
    }
    metrics_path = os.path.splitext(args.model_out)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {args.model_out}")
    print(f"Saved metrics to {metrics_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())