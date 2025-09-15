
import os
import argparse
import pandas as pd
from datetime import date
from typing import Optional, List

def fetch_fred(series: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    end = end or str(date.today())
    api_key = os.getenv("FRED_API_KEY", None)
    df_all = pd.DataFrame()
    try:
        if api_key:
            from fredapi import Fred
            fred = Fred(api_key=api_key)
            for s in series:
                ser = fred.get_series(s, observation_start=start, observation_end=end)
                df_all[s] = ser
        else:
            from pandas_datareader import data as pdr
            for s in series:
                ser = pdr.DataReader(s, "fred", start, end)
                df_all[s] = ser[s]
    except Exception as e:
        raise RuntimeError(f"Error fetching FRED data: {e}")
    df_all.index.name = "date"
    df_all = df_all.sort_index()
    return df_all

def cli():
    parser = argparse.ArgumentParser(description="Download FRED series and save as parquet/csv")
    parser.add_argument("--config", required=True, help="Path to configs/features.yaml")
    parser.add_argument("--out", required=True, help="Output file path (.parquet or .csv)")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    start = cfg.get("start_date", "2015-01-01")
    end = cfg.get("end_date", None)
    series = cfg.get("series", [])
    df = fetch_fred(series, start, end)
    df = df.ffill().bfill()

    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if out.endswith(".parquet"):
        df.to_parquet(out)
    else:
        df.to_csv(out, index=True)

if __name__ == "__main__":
    cli()
