#!/usr/bin/env python3
"""
Load a trained model and produce predictions in sample submission format.

Usage:
    python3 scripts/predict.py --model ./models/poc_model.pkl --test ./data/test.csv --out ./submissions/predictions.csv
"""
import argparse
import os
import pandas as pd
import joblib
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Produce predictions with saved model")
    p.add_argument("--model", "-m", default="./models/poc_model.pkl", help="Path to saved model (joblib)")
    p.add_argument("--test", "-t", default="./data/test.csv", help="Path to test CSV")
    p.add_argument("--out", "-o", default="./submissions/predictions.csv", help="Path to write predictions CSV")
    return p.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}", flush=True)
        return 2
    if not os.path.exists(args.test):
        print(f"Test file not found: {args.test}", flush=True)
        return 3

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"Loading model from {args.model} ...")
    model = joblib.load(args.model)

    print(f"Reading test data from {args.test} ...")
    df = pd.read_csv(args.test)

    ids = df["id"] if "id" in df.columns else pd.Series(range(len(df)), name="id")

    # Prepare features: select numeric columns and drop id if present
    X = df.select_dtypes(include=[np.number]).copy()
    if "id" in X.columns:
        X = X.drop(columns=["id"])

    X = X.fillna(X.median())

    print(f"Predicting {len(X)} rows ...")
    preds = model.predict(X)
    preds = np.clip(preds, 0, None)

    out_df = pd.DataFrame({"id": ids, "Calories": preds})
    out_df.to_csv(args.out, index=False)
    print(f"Wrote predictions to {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())