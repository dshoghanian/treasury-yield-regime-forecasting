
import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, in_dim: int, hidden=[256, 128, 64], out_dim=1, p=0.3):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.BatchNorm1d(h), nn.LeakyReLU(), nn.Dropout(p)]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last, out_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        return self.head(self.backbone(x))
