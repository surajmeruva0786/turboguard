"""Direct-regression RUL estimation via the hybrid model's RUL head (README 12.1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.turboguard_hybrid import TurboGuardHybrid

DEFAULT_RUL_CAP = 125.0


def piecewise_linear_rul_target(rul_cycles: float, cap: float = DEFAULT_RUL_CAP) -> float:
    """Cap the RUL regression target at ``cap`` cycles in the healthy regime.

    Discourages the model from confidently predicting arbitrarily large RUL
    values far from failure, where the signal carries little information
    about exact remaining life anyway.
    """
    return float(min(rul_cycles, cap))


def load_direct_regression_model(
    checkpoint_path: str | Path, n_channels: int = 3, n_classes: int = 5, **model_kwargs
) -> TurboGuardHybrid:
    model = TurboGuardHybrid(n_channels=n_channels, n_classes=n_classes, **model_kwargs)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_rul(model: TurboGuardHybrid, windows: np.ndarray | torch.Tensor) -> np.ndarray:
    """Predict RUL for a batch of window sequences, shape ``(B, K, n_channels, n_samples)``."""
    if not isinstance(windows, torch.Tensor):
        windows = torch.from_numpy(np.asarray(windows, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        _, rul_pred = model(windows)
    return rul_pred.numpy()
