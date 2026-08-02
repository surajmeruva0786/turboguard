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
    predictions (``d_i > 0``) more heavily than early ones.

    Designed for PHM-2012-scale errors (tens of cycles); on RUL targets
    with a much larger native range (e.g. real IMS's ~1000-snapshot
    trajectories) a single bad prediction can overflow ``exp`` to ``inf``,
    which isn't valid JSON. The exponent is clipped to keep the score a
    large-but-finite float instead — still clearly a bad score, just
    representable.
    """
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    d = y_pred - y_true
    exponent = np.where(d < 0, -d / 13, d / 10)
    scores = np.exp(np.minimum(exponent, 700.0)) - 1
    return float(np.sum(scores))


def rul_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE/MAPE/PHM score, computed over only the finite predictions.

    The degradation model (src/rul/degradation_model.py) deliberately
    returns NaN when a trajectory isn't fit for extrapolation (e.g. a
    non-increasing health-indicator window) rather than guessing — correct
    per-point behaviour, but a single NaN would otherwise poison every
    aggregate metric here (sqrt(mean(...NaN...)) == NaN) and hide how many
    of the *other* predictions were actually fine. n_valid/n_dropped make
    that visible instead.
    """
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    finite = np.isfinite(y_pred)
    n_total = len(y_pred)
    n_valid = int(finite.sum())
    y_true_valid, y_pred_valid = y_true[finite], y_pred[finite]
    return {
        "rmse": rmse(y_true_valid, y_pred_valid) if n_valid else float("nan"),
        "mape": mape(y_true_valid, y_pred_valid) if n_valid else float("nan"),
        "phm_score": phm_score(y_true_valid, y_pred_valid) if n_valid else float("nan"),
        "n_samples": n_total,
        "n_valid": n_valid,
        "n_dropped_nonfinite": n_total - n_valid,
    }
