"""Feature-space autoencoder for the health-indicator RUL approach (README 12.2).

Trained on *healthy*-only engineered feature vectors; reconstruction error
on unseen windows serves as a Health Indicator (HI) that rises as the
bearing degrades away from its healthy manifold.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureAutoencoder(nn.Module):
    """Symmetric MLP autoencoder over the ~176-dim engineered feature vector."""

    def __init__(self, input_dim: int = 176, latent_dim: int = 16, hidden_dims: tuple[int, ...] = (64, 32)):
        super().__init__()
        encoder_layers: list[nn.Module] = []
        dims = [input_dim, *hidden_dims]
        for in_dim, out_dim in zip(dims, dims[1:]):
            encoder_layers += [nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True)]
        encoder_layers.append(nn.Linear(dims[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        rev_dims = [latent_dim, *reversed(hidden_dims)]
        for in_dim, out_dim in zip(rev_dims, rev_dims[1:]):
            decoder_layers += [nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True)]
        decoder_layers.append(nn.Linear(rev_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean-squared reconstruction error, i.e. the raw Health Indicator."""
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=-1)
