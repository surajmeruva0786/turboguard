"""Waveform augmentations applied during deep-model training (README 10, "Augmentation").

All functions operate on a batch of raw windows, shape ``(B, n_channels,
n_samples)`` (or ``(B, K, n_channels, n_samples)`` for the hybrid model's
sequence input, applied identically across the ``K`` axis), and are
no-ops at eval time — callers should only invoke these on training batches.
"""
from __future__ import annotations

import torch


def time_shift(x: torch.Tensor, max_shift_pct: float = 0.05) -> torch.Tensor:
    """Circularly shift each sample along the time axis by up to ``max_shift_pct``."""
    n_samples = x.shape[-1]
    max_shift = max(1, int(n_samples * max_shift_pct))
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    return torch.roll(x, shifts=shift, dims=-1)


def amplitude_scale(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """Scale each sample by a factor drawn from ``N(1, std**2)``."""
    shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    factors = 1.0 + std * torch.randn(shape, device=x.device, dtype=x.dtype)
    return x * factors


def add_gaussian_noise(x: torch.Tensor, snr_db: float = 30.0) -> torch.Tensor:
    """Add Gaussian noise at the given signal-to-noise ratio (per sample)."""
    signal_power = x.pow(2).mean(dim=tuple(range(1, x.dim())), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(x) * noise_power.sqrt()
    return x + noise


def augment_batch(
    x: torch.Tensor,
    time_shift_pct: float = 0.05,
    amplitude_scale_std: float = 0.05,
    noise_snr_db: float = 30.0,
) -> torch.Tensor:
    """Apply the full augmentation pipeline (time shift -> amplitude scale -> noise)."""
    x = time_shift(x, time_shift_pct)
    x = amplitude_scale(x, amplitude_scale_std)
    x = add_gaussian_noise(x, noise_snr_db)
    return x
