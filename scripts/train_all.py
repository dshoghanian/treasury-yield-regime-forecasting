
#!/usr/bin/env python
import subprocess, sys
for model in ["fnn", "cnn", "lstm", "transformer"]:
    print(f"=== Training {model} ===")
    cmd = [sys.executable, "-m", "treasury_forecaster.training.train",
           "--config", "configs/base.yaml",
           "--data", "data/processed/features.parquet",
           "--model", model]
    subprocess.run(cmd, check=True)
