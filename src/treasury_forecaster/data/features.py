
import argparse
import os
from typing import Optional
import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame, cfg_base: dict) -> pd.DataFrame:
    df = df.copy()

    # Autoregressive
    if "DGS10" in df.columns:
        df["DGS10_lag1"] = df["DGS10"].shift(1)

    # Spreads
    if {"DGS10", "DGS2"}.issubset(df.columns):
        df["T10Y2Y"] = df["DGS10"] - df["DGS2"]
    if {"DGS10", "DGS3MO"}.issubset(df.columns):
        df["T10Y3M"] = df["DGS10"] - df["DGS3MO"]

    # MAs, momentum, volatility
    if "DGS10" in df.columns:
        for w in [5, 20, 50, 100]:
            df[f"MA_{w}_DGS10"] = df["DGS10"].rolling(w).mean()
        for roc in [5, 20]:
            df[f"ROC_{roc}_DGS10"] = df["DGS10"].pct_change(roc)
        df["Volatility_20_DGS10"] = df["DGS10"].pct_change().rolling(20).std()

    # Simple regime labels via EFFR changes
    regimes_cfg = cfg_base.get("regimes", {}).get("monetary_policy", {})
    series = regimes_cfg.get("series", "EFFR")
    lb = regimes_cfg.get("lookback_days", 60)
    if series in df.columns:
        chg = df[series].diff(lb)
        df["regime_monetary_policy"] = np.select(
            [
                chg <= regimes_cfg.get("thresholds", {}).get("easing", -0.05),
                chg >= regimes_cfg.get("thresholds", {}).get("tightening", 0.05)
            ],
            ["easing", "tightening"],
            default="stable"
        )

    df = df.dropna().copy()
    return df

def cli():
    parser = argparse.ArgumentParser(description="Build engineered features from raw data")
    parser.add_argument("--config", required=True, help="Path to configs/base.yaml")
    parser.add_argument("--features", required=True, help="Path to configs/features.yaml")
    parser.add_argument("--in", dest="in_path", required=True, help="Input raw file (.parquet or .csv)")
    parser.add_argument("--out", required=True, help="Output processed file (.parquet or .csv)")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg_base = yaml.safe_load(f)

    # Load data
    if args.in_path.endswith(".parquet"):
        df = pd.read_parquet(args.in_path)
    else:
        tmp = pd.read_csv(args.in_path)
        date_col = "date" if "date" in tmp.columns else ("DATE" if "DATE" in tmp.columns else None)
        if date_col is None:
            raise ValueError("Expected a date column named 'date' or 'DATE' in the CSV.")
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        df = tmp.sort_values(date_col).set_index(date_col)
        df.index.name = "date"

    df_feat = build_features(df, cfg_base)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.out.endswith(".parquet"):
        df_feat.to_parquet(args.out)
    else:
        df_feat.to_csv(args.out, index=True)

if __name__ == "__main__":
    cli()
