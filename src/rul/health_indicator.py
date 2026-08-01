"""Health-indicator estimation via a feature-space autoencoder (README 12.2, stage 1).

The autoencoder is trained on healthy-only engineered feature vectors;
reconstruction error on unseen windows is the raw Health Indicator (HI) —
it rises as a window's features drift away from the healthy manifold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.models.autoencoder import FeatureAutoencoder
from src.utils.seed import set_seed


@dataclass
class HealthIndicatorModel:
    """A fitted autoencoder + the healthy-population feature normalisation
    it was trained with (must be reapplied at inference time)."""

    autoencoder: FeatureAutoencoder
    feature_mean: np.ndarray
    feature_std: np.ndarray

    def compute(self, X: np.ndarray, clip: float = 1e4) -> np.ndarray:
        """Health indicator (reconstruction error) for each row of ``X``, shape ``(n,)``.

        ``clip`` bounds the raw MSE: with a tiny healthy-only training set,
        reconstruction error on a severely out-of-distribution (heavily
        degraded) input can extrapolate to an unbounded value that carries
        no additional signal beyond "very unhealthy" and would otherwise
        risk float overflow in the downstream degradation-curve fit.
        """
        X_norm = (np.asarray(X, dtype=np.float64) - self.feature_mean) / self.feature_std
        x = torch.from_numpy(X_norm.astype(np.float32))
        self.autoencoder.eval()
        with torch.no_grad():
            err = self.autoencoder.reconstruction_error(x)
        return np.clip(err.numpy(), 0.0, clip)


def fit_health_indicator_model(
    X_healthy: np.ndarray,
    latent_dim: int = 16,
    hidden_dims: tuple[int, ...] = (64, 32),
    epochs: int = 300,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    seed: int = 42,
) -> HealthIndicatorModel:
    """Fit a :class:`FeatureAutoencoder` on healthy-only feature vectors.

    ``X_healthy`` has shape ``(n_healthy_windows, n_features)``. Features are
    standardised (z-score, healthy-population stats) before fitting so the
    reconstruction-error scale is comparable across differently-scaled
    engineered features. ``weight_decay`` regularises the (typically
    heavily overparameterised relative to a small healthy-only sample set)
    autoencoder so reconstruction error on far out-of-distribution
    (degraded) inputs stays a meaningful monitoring signal instead of
    exploding from unconstrained extrapolation.
    """
    set_seed(seed)
    X_healthy = np.asarray(X_healthy, dtype=np.float64)
    mean = X_healthy.mean(axis=0)
    std = X_healthy.std(axis=0)
    std[std < 1e-8] = 1e-8
    X_norm = (X_healthy - mean) / std

    model = FeatureAutoencoder(input_dim=X_healthy.shape[1], latent_dim=latent_dim, hidden_dims=hidden_dims)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x = torch.from_numpy(X_norm.astype(np.float32))

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = model.reconstruction_error(x).mean()
        loss.backward()
        optimizer.step()

    return HealthIndicatorModel(autoencoder=model, feature_mean=mean, feature_std=std)
