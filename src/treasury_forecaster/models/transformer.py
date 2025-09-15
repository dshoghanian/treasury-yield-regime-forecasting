
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, in_feat: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128, out_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_feat, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim))

    def forward(self, x):
        z = self.input_proj(x)
        z = self.encoder(z)
        return self.head(z[:, -1, :])
