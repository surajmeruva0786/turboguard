"""1D-CNN encoder and TurboGuard-CNN fault classifier (README section 9.2).

:class:`CNNEncoder` is factored out so :mod:`turboguard_hybrid` can reuse
the identical time-distributed feature extractor feeding its Bi-LSTM.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """4-block 1D-CNN feature extractor: (B, n_channels, n_samples) -> (B, 256)."""

    def __init__(self, n_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=64, stride=8),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=32, stride=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=16, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=8),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    @property
    def output_dim(self) -> int:
        return 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.pool(x).squeeze(-1)


class TurboGuardCNN(nn.Module):
    """Single-window fault classifier. Returns raw logits (apply softmax/
    ``CrossEntropyLoss`` externally)."""

    def __init__(self, n_channels: int = 3, n_classes: int = 5, dropout: float = 0.4):
        super().__init__()
        self.encoder = CNNEncoder(n_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.dropout(self.encoder(x))
        return self.classifier(feats)
