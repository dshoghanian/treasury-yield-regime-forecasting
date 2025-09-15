
#!/usr/bin/env python
import os, numpy as np, pandas as pd
np.random.seed(42)
n = 300
dates = pd.date_range("2023-01-01", periods=n, freq="B")
dgs10 = 3.0 + 0.1*np.sin(np.linspace(0, 6.28, n)) + 0.2*np.random.randn(n)
dgs2 = dgs10 - 0.5 + 0.05*np.random.randn(n)
effr = 5.25 + 0.02*np.random.randn(n)
cpi = 300 + np.cumsum(0.05 + 0.2*np.random.randn(n))
spx = 4000 + np.cumsum(np.random.randn(n))
df = pd.DataFrame({"DGS10": dgs10, "DGS2": dgs2, "EFFR": effr, "CPIAUCSL": cpi, "SP500": spx}, index=dates)
df.index.name = "date"
os.makedirs("data/processed", exist_ok=True)
df.to_parquet("data/processed/features.parquet")
print("Wrote synthetic features to data/processed/features.parquet")
