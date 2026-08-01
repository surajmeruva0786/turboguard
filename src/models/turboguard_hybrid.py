"""TurboGuard-Hybrid: CNN + Bi-LSTM multi-task model (README section 9.3).

Consumes a sequence of ``K`` consecutive 1-second windows and produces
both a fault-class prediction (mean-pooled over time) and a RUL
regression (from the last timestep) from one shared encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.turboguard_cnn import CNNEncoder


class TurboGuardHybrid(nn.Module):
    """Multi-task fault classification + RUL regression.

    Input: ``(B, K, n_channels, n_samples)``.
    Outputs: ``(fault_logits, rul_pred)`` of shapes ``(B, n_classes)`` and ``(B,)``.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.encoder = CNNEncoder(n_channels)
        self.lstm = nn.LSTM(
            input_size=self.encoder.output_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        lstm_out_dim = lstm_hidden * 2
        self.dropout = nn.Dropout(dropout)
        self.fault_head = nn.Linear(lstm_out_dim, n_classes)
        self.rul_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, n_channels, n_samples = x.shape
        x = x.view(batch * seq_len, n_channels, n_samples)
        feats = self.encoder(x)
        feats = feats.view(batch, seq_len, -1)

        lstm_out, _ = self.lstm(feats)
        lstm_out = self.dropout(lstm_out)

        fault_logits = self.fault_head(lstm_out.mean(dim=1))
        rul_pred = self.rul_head(lstm_out[:, -1, :]).squeeze(-1)
        return fault_logits, rul_pred
