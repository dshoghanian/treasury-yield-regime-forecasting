
import torch
import torch.nn as nn

class TemporalCNN(nn.Module):
    def __init__(self, in_feat: int, seq_len: int, out_dim=1, channels=32):
        super().__init__()
        self.conv2 = nn.Conv1d(in_feat, channels, kernel_size=2)
        self.conv3 = nn.Conv1d(in_feat, channels, kernel_size=3)
        self.conv5 = nn.Conv1d(in_feat, channels, kernel_size=5)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels*3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, F, L)
        c2 = self.pool(torch.relu(self.conv2(x)))
        c3 = self.pool(torch.relu(self.conv3(x)))
        c5 = self.pool(torch.relu(self.conv5(x)))
        z = torch.cat([c2, c3, c5], dim=1)
        return self.head(z)
