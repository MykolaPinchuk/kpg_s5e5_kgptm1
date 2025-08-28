#!/usr/bin/env python3
"""Create a header-preserving random subsample of a CSV file."""
import argparse
import pandas as pd
import os
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Create header-preserving random subsample")
    p.add_argument("--input", "-i", default="./data/train.csv", help="Path to input train.csv")
    p.add_argument("--output", "-o", default="./data/subsample_train.csv", help="Path to output subsample")
    p.add_argument("--n", "-n", type=int, default=10000, help="Number of sample rows (including header not counted)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"Reading {args.input} ...")
    df = pd.read_csv(args.input)
    total = len(df)
    if args.n >= total:
        print(f"Requested sample size {args.n} >= rows in input ({total}). Writing full file.")
        df.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")
        return 0

    sampled = df.sample(n=args.n, random_state=args.seed)
    sampled.to_csv(args.output, index=False)
    print(f"Wrote subsample with {len(sampled)} rows to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())