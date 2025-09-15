
import argparse, os
import pandas as pd

def cli():
    ap = argparse.ArgumentParser(description="Ingest an existing CSV of macro/market features and save as processed Parquet/CSV.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output file (.parquet or .csv)")
    ap.add_argument("--date-col", default="DATE", help="Name of the date column (default: DATE)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    if args.date_col not in df.columns:
        raise ValueError(f"Date column '{args.date_col}' not found.")
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    df = df.sort_values(args.date_col).set_index(args.date_col)
    df.index.name = "date"
    df = df.ffill().bfill()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    if args.out_path.endswith(".parquet"):
        df.to_parquet(args.out_path)
    else:
        df.to_csv(args.out_path, index=True)

    print(f"[treasury-ingest] wrote {args.out_path}  rows={len(df)} cols={df.shape[1]}")
