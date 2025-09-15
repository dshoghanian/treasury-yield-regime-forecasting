
import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

from treasury_forecaster.models.fnn import FNN
from treasury_forecaster.models.cnn import TemporalCNN
from treasury_forecaster.models.lstm import LSTMModel
from treasury_forecaster.models.transformer import TransformerModel

class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target: str, seq_len: int):
        self.df = df.copy()
        self.target = target
        self.seq_len = seq_len
        self.X, self.y = self._make_windows()

    def _make_windows(self):
        values = self.df.values
        tgt_idx = list(self.df.columns).index(self.target)
        X_list, y_list = [], []
        for i in range(len(values) - self.seq_len):
            X_list.append(values[i:i+self.seq_len, :])
            y_list.append(values[i+self.seq_len, tgt_idx])
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def build_model(model_name: str, in_feat: int, seq_len: int):
    if model_name == "fnn":
        return FNN(in_dim=in_feat*seq_len, out_dim=1)
    if model_name == "cnn":
        return TemporalCNN(in_feat=in_feat, seq_len=seq_len, out_dim=1)
    if model_name == "lstm":
        return LSTMModel(in_feat=in_feat, hidden=64, layers=2, bidirectional=True, out_dim=1)
    if model_name == "transformer":
        return TransformerModel(in_feat=in_feat, d_model=64, nhead=4, num_layers=2, out_dim=1)
    raise ValueError(f"Unknown model: {model_name}")

def train_epoch(model, loader, optim, loss_fn, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        if xb.dim() == 3 and isinstance(model, FNN):
            xb = xb.reshape(xb.size(0), -1)
        optim.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optim.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    preds, trues = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
        if xb.dim() == 3 and isinstance(model, FNN):
            xb = xb.reshape(xb.size(0), -1)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total += loss.item() * xb.size(0)
        preds.append(pred.squeeze(1).cpu().numpy())
        trues.append(yb.squeeze(1).cpu().numpy())
    import numpy as np
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    rmse = mean_squared_error(trues, preds, squared=False)
    mae = mean_absolute_error(trues, preds)
    return total / len(loader.dataset), rmse, mae

def main():
    parser = argparse.ArgumentParser(description="Train regime-aware models for 10Y yield")
    parser.add_argument("--config", required=True, help="Path to base config YAML")
    parser.add_argument("--data", required=True, help="Processed features file (.parquet or .csv)")
    parser.add_argument("--model", default=None, help="Model override: fnn/cnn/lstm/transformer")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed); np.random.seed(seed)

    target = cfg.get("target", "DGS10")
    seq_len = cfg.get("sequence_length", 10)
    model_name = args.model or cfg.get("model", "lstm")
    batch_size = cfg.get("batch_size", 64)
    lr = cfg.get("learning_rate", 1e-3)
    epochs = cfg.get("epochs", 50)

    # Load data
    if args.data.endswith(".parquet"):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data, parse_dates=["date"], index_col="date")

    # Chronological split
    n = len(df)
    test_size = int(n * cfg.get("test_size", 0.15))
    val_size = int(n * cfg.get("val_size", 0.15))
    train_end = n - (test_size + val_size)
    val_end = n - test_size
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end - seq_len:val_end]
    df_test = df.iloc[val_end - seq_len:]

    # Standardize
    feat_cols = df.columns.tolist()
    mean = df_train.mean()
    std = df_train.std().replace(0, 1.0)
    df_train = (df_train - mean) / std
    df_val = (df_val - mean) / std
    df_test = (df_test - mean) / std

    # Datasets
    ds_train = SeqDataset(df_train[feat_cols], target, seq_len)
    ds_val = SeqDataset(df_val[feat_cols], target, seq_len)
    ds_test = SeqDataset(df_test[feat_cols], target, seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, in_feat=len(feat_cols), seq_len=seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(ds_val, batch_size=batch_size)
    te_loader = DataLoader(ds_test, batch_size=batch_size)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs+1):
        tr_loss = train_epoch(model, tr_loader, opt, loss_fn, device)
        va_loss, va_rmse, va_mae = eval_epoch(model, va_loader, loss_fn, device)
        if va_loss < best_val:
            best_val, best_state = va_loss, {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:03d} | Train {tr_loss:.6f} | Val {va_loss:.6f} (RMSE {va_rmse:.4f} MAE {va_mae:.4f})")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    te_loss, te_rmse, te_mae = eval_epoch(model, te_loader, loss_fn, device)
    print(f"Test Loss {te_loss:.6f} | Test RMSE {te_rmse:.4f} | Test MAE {te_mae:.4f}")

if __name__ == "__main__":
    main()
