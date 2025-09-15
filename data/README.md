
# Data Directory

This repository does not commit large datasets. You have two options:

**Option A — Rebuild from FRED (no local CSV):**
```bash
treasury-download --config configs/features.yaml --out data/raw/fred.parquet
treasury-features --config configs/base.yaml --features configs/features.yaml   --in data/raw/fred.parquet --out data/processed/features.parquet
```

**Option B — Use your combined CSV (recommended if you already have one):**
```bash
treasury-ingest --in examples/fred_combined_10yrs.csv --out data/processed/features.parquet --date-col DATE
```
