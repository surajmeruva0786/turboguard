"""RUL regression metrics: RMSE, MAPE, and the PHM 2012 asymmetric score (README 11.3)."""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Absolute Percentage Error. ``eps`` avoids division by zero at
    the exact failure point (``y_true == 0``)."""
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs((y_pred - y_true) / np.maximum(np.abs(y_true), eps))) * 100)


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PHM 2012 asymmetric scoring function (README 11.3): penalises late
    predictions (``d_i > 0``) more heavily than early ones."""
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


def rul_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "phm_score": phm_score(y_true, y_pred),
        "n_samples": int(len(np.asarray(y_true))),
    }
