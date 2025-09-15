
# Regime-Aware Neural Networks for 10Y U.S. Treasury Yield Forecasting

**Summary:** Deep learning models (FNN, CNN, LSTM, Transformer) with monetary policy regime indicators to forecast the 10-Year U.S. Treasury yield, plus shock simulations and feature importance for actionable insights.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip && pip install -r requirements.txt

# Use your own CSV (no download required)
treasury-ingest --in examples/fred_combined_10yrs.csv --out data/processed/features.parquet --date-col DATE

# Or rebuild from FRED
# export FRED_API_KEY="your_api_key"  # optional
# treasury-download --config configs/features.yaml --out data/raw/fred.parquet
# treasury-features --config configs/base.yaml --features configs/features.yaml --in data/raw/fred.parquet --out data/processed/features.parquet

# Train a model (choices: fnn, cnn, lstm, transformer)
python -m treasury_forecaster.training.train --config configs/base.yaml --data data/processed/features.parquet --model lstm

# Explore the notebook
jupyter notebook notebooks/treasury_yield_forecasting_regime_nn.ipynb
```

## Structure
```
├── configs/ (base.yaml, features.yaml)
├── src/treasury_forecaster/
│   ├── data/ (fred_loader.py, features.py, ingest_existing.py, simulate_shocks.py)
│   ├── models/ (fnn.py, cnn.py, lstm.py, transformer.py, domain_informed.py)
│   ├── training/train.py
│   └── analysis/feature_importance.py
├── scripts/ (ingest_existing_csv.py, download_data.py, generate_sample_data.py, train_all.py)
├── examples/ (fred_combined_10yrs.csv, README.md)
├── notebooks/ (treasury_yield_forecasting_regime_nn.ipynb)
├── docs/ (Final_Project_Report.docx, Presentation.pptx)
└── ...
```

## Notes
- Regime labels default to EFFR change over a 60-day window; swap with your preferred regime detector if needed.
- The training pipeline standardizes features based on the train split and supports multi-horizon targets.
- Shock simulations and feature importance have ready hooks to extend.

## License
MIT
