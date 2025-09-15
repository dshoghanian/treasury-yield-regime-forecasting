
# Regime‑Aware Neural Networks for 10‑Year U.S. Treasury Yield Forecasting

> Deep learning models (FNN, CNN, LSTM, Transformer) augmented with **monetary‑policy regime indicators** to forecast the 10‑Year U.S. Treasury yield. Includes a reproducible data pipeline (use your **CSV** or rebuild from **FRED**), simple **shock simulations**, and hooks for **feature importance**.


## Why this matters

The 10‑Year U.S. Treasury yield (DGS10) is the anchor for mortgage rates, corporate borrowing costs, and the discount rate used across equity valuation and project finance. A regime‑aware approach helps the model behave differently in **tightening**, **stable**, and **easing** policy environments, improving robustness during structural shifts (like post‑pandemic inflation and rapid hiking cycles).


## What’s included

- **CSV‑first workflow**: Ingest your combined dataset with `treasury-ingest` (no external downloads required).
- **FRED rebuild path**: Optional pipeline to fetch raw series and re‑create features end‑to‑end.
- **Feature engineering**: yield‑curve spreads (10Y–2Y, 10Y–3M), trend/momentum, volatility, and policy regimes.
- **Neural architectures**: FNN, Temporal CNN, LSTM, and Transformer (PyTorch).
- **Training loop**: chronological splits; standardized features; RMSE/MAE reporting.
- **Shock utilities**: apply simple additive shocks to key drivers (e.g., EFFR, CPI) to sanity‑check sensitivities.
- **Portfolio‑ready repo**: clean structure, minimal CI, MIT license, and docs folder for your report/slides.


## Data options

You can either ingest your own CSV (recommended for demos) or rebuild from FRED.

### A) Use your CSV (no downloads)
The repository ships with `examples/fred_combined_10yrs.csv` (you can replace it with your own).

```bash
# Create & activate environment
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip && pip install -r requirements.txt

# Ingest your CSV -> clean, time-indexed file
treasury-ingest --in examples/fred_combined_10yrs.csv   --out data/processed/features.parquet --date-col DATE
```

### B) Rebuild from FRED (reproducible pipeline)
```bash
# Optional (faster, higher limits)
# export FRED_API_KEY="your_api_key"

treasury-download --config configs/features.yaml --out data/raw/fred.parquet

treasury-features  --config configs/base.yaml --features configs/features.yaml   --in data/raw/fred.parquet --out data/processed/features.parquet
```


## Quickstart (train & explore)

```bash
# Train a model (choices: fnn | cnn | lstm | transformer)
python -m treasury_forecaster.training.train --config configs/base.yaml   --data data/processed/features.parquet --model lstm

# Explore the analysis notebook
jupyter notebook notebooks/treasury_yield_forecasting_regime_nn.ipynb
```


## Configuration at a glance

Edit `configs/base.yaml` to control key behavior:

```yaml
seed: 42
target: DGS10              # prediction target
sequence_length: 10        # lookback window in days
test_size: 0.15            # chronological split
val_size: 0.15
batch_size: 64
epochs: 50
learning_rate: 0.001
model: lstm                # fnn | cnn | lstm | transformer

regimes:
  monetary_policy:
    method: "thresholds"
    series: "EFFR"
    thresholds:
      easing: -0.05        # <= -5 bps over lookback
      tightening: 0.05     # >= +5 bps over lookback
    lookback_days: 60
```

- **Change targets**: set `target: DGS2` (or any column in your dataset).  
- **Switch models**: pass `--model fnn|cnn|lstm|transformer` at train time.  
- **Windowing**: adjust `sequence_length` to trade off recency vs. context.


## Feature engineering (implemented)

- **Autoregressive**: `DGS10_lag1`  
- **Yield‑curve spreads**: `T10Y2Y = DGS10 - DGS2`, `T10Y3M = DGS10 - DGS3MO`  
- **Trend & momentum**: `MA_{5,20,50,100}_DGS10`, `ROC_{5,20}_DGS10`  
- **Volatility**: rolling 20‑day volatility of DGS10  

These are computed in `src/treasury_forecaster/data/features.py`. You can extend this to include curvature (2s10s30s butterfly), forward rates, RSI/MACD, or term‑premium proxies.


## Regime labeling

A simple, transparent regime tag is created from **EFFR** changes over a moving window. Defaults (in `base.yaml`) classify *easing*, *stable*, or *tightening* based on ±5 bps over 60 days. Advanced regime models (e.g., HMM, yield‑curve states, volatility regimes) can be plugged in via `data/regimes.py`.


## Models (PyTorch)

- **FNN**: MLP on flattened windows (BatchNorm + Dropout).  
- **Temporal CNN**: parallel 1D kernels (2/3/5) → pooled → dense head.  
- **LSTM**: bidirectional LSTM; last hidden state → regression head.  
- **Transformer**: encoder with learned input projection; last token head.  

Swap architectures via `--model`, keep the rest identical (data, splits, metrics).


## Training & evaluation

- Time‑ordered split into **train / validation / test** (validation/test start with overlap so sequence windows align).  
- Standardization (mean/std) learned on **train** and applied to val/test.  
- Metrics reported: **MSE (loss)** + **RMSE/MAE** for readability.  
- Try all models quickly:

```bash
python scripts/train_all.py
```


## Economic shock simulation (sanity checks)

```python
from treasury_forecaster.data.simulate_shocks import apply_shock
import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")

# Example: +50 bps to EFFR and +50 bps to CPI
df_shocked = apply_shock(df, {"EFFR": 0.50, "CPIAUCSL": 0.50})

# Standardize with your train stats and run model inference to compare deltas
```

This simple utility applies **additive** shocks to chosen series so you can gauge directional sensitivity before/after training or by regime.


## Results highlights (from the included report/slides)

- **Enhanced LSTM** reached **R² ≈ 0.8769** with strong level accuracy.  
- A **domain‑informed hybrid** achieved **R² ≈ 0.9125**, outperforming baseline DL models.  
- **Directional accuracy** peaked near **~35%**; **Bayesian LSTM** provided well‑calibrated intervals.  
- Shock tests indicated 10Y yields were **more sensitive to CPI surprises** than to equal‑sized Fed‑funds shocks.  

> Full methodology, comparisons, and figures are in `docs/Final_Project_Report.docx` and `docs/Presentation.pptx`.
> The codebase here implements the core pipeline (FNN/CNN/LSTM/Transformer + basic regimes). Advanced variants can be layered on as extensions.


## Repository structure (abridged)

```
├── configs/ (base.yaml, features.yaml)
├── src/treasury_forecaster/
│   ├── data/ (fred_loader.py, features.py, ingest_existing.py, simulate_shocks.py, regimes.py)
│   ├── models/ (fnn.py, cnn.py, lstm.py, transformer.py, domain_informed.py)
│   ├── training/train.py
│   └── analysis/feature_importance.py
├── scripts/ (ingest_existing_csv.py, download_data.py, generate_sample_data.py, train_all.py)
├── examples/ (fred_combined_10yrs.csv)
├── notebooks/ (treasury_yield_forecasting_regime_nn.ipynb)
├── docs/ (Final_Project_Report.docx, Presentation.pptx)
└── data/ (ignored: raw/interim/processed)
```


## License & attribution

- **License:** MIT (see `LICENSE`).  
- **Data attribution:** Federal Reserve Economic Data (**FRED**).  
- **Citation:** see `CITATION.cff` or cite the report/slides in `docs/`.


## Disclaimer

This repository is for **educational and research** purposes only and does **not** constitute financial advice or a solicitation to transact. Always perform your own due diligence.
