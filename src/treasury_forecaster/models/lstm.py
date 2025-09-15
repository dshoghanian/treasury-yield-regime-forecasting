
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, in_feat: int, hidden: int = 64, layers: int = 2, bidirectional: bool = True, out_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_feat, hidden_size=hidden, num_layers=layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout)
        mult = 2 if bidirectional else 1
        self.head = nn.Sequential(nn.LayerNorm(hidden*mult), nn.Linear(hidden*mult, out_dim))

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)
