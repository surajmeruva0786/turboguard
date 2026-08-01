"""Combine direct-regression and health-indicator RUL estimates (README 12.2, final step)."""
from __future__ import annotations

import numpy as np


def combine_rul_estimates(
    direct_pred: np.ndarray | float, hi_pred: np.ndarray | float, weight_direct: float = 0.5
) -> np.ndarray | float:
    """Weighted average of the two RUL estimates."""
    return weight_direct * np.asarray(direct_pred) + (1 - weight_direct) * np.asarray(hi_pred)


def learn_combination_weight(
    direct_preds: np.ndarray, hi_preds: np.ndarray, actual: np.ndarray, n_grid: int = 101
) -> float:
    """Grid-search the blend weight (in ``[0, 1]``) minimising validation MSE."""
    direct_preds = np.asarray(direct_preds, dtype=np.float64)
    hi_preds = np.asarray(hi_preds, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)

    weights = np.linspace(0.0, 1.0, n_grid)
    best_w, best_mse = 0.5, float("inf")
    for w in weights:
        combined = w * direct_preds + (1 - w) * hi_preds
        mse = float(np.mean((combined - actual) ** 2))
        if mse < best_mse:
            best_mse, best_w = mse, float(w)
    return best_w
