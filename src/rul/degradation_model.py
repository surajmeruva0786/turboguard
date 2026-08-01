"""Exponential degradation model fit to a Health-Indicator trajectory (README 12.2, stage 2).

``HI(t) = a * exp(b * t)``, extrapolated to a configurable failure
threshold to produce a RUL estimate.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit


def _exp_model(t: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.exp(b * t)


def fit_exponential_degradation(
    time_index: np.ndarray, hi_values: np.ndarray, floor: float = 1e-6
) -> tuple[float, float]:
    """Fit ``a, b`` in ``HI(t) = a * exp(b*t)`` via nonlinear least squares.

    ``hi_values`` are clipped to ``floor`` before fitting since reconstruction
    error is expected to be non-negative but can be exactly zero for
    near-perfect reconstructions.
    """
    time_index = np.asarray(time_index, dtype=np.float64)
    hi_safe = np.clip(np.asarray(hi_values, dtype=np.float64), floor, None)
    a0 = max(float(hi_safe[0]), floor)
    b0 = 0.05
    try:
        (a, b), _ = curve_fit(_exp_model, time_index, hi_safe, p0=[a0, b0], maxfev=10000)
    except RuntimeError:
        a, b = a0, b0
    return float(a), float(b)


def predict_time_to_threshold(a: float, b: float, threshold: float, current_time: float = 0.0) -> float:
    """Extrapolate ``HI(t) = a*exp(b*t)`` to ``threshold``; return remaining time (>= 0).

    Returns ``nan`` if the trajectory is not increasing (``b <= 0``) or has
    already crossed the threshold in a way the model can't invert (``a`` or
    ``threshold`` non-positive) — degradation cannot be extrapolated in
    either case.
    """
    if b <= 0 or a <= 0 or threshold <= 0:
        return float("nan")
    t_fail = math.log(threshold / a) / b
    return max(t_fail - current_time, 0.0)


def estimate_rul_from_trajectory(
    time_index: np.ndarray,
    hi_values: np.ndarray,
    threshold: float,
    current_time: float | None = None,
) -> float:
    """Fit the degradation model to the observed trajectory and extrapolate
    to ``threshold``, evaluated at ``current_time`` (defaults to the last
    observed time index)."""
    a, b = fit_exponential_degradation(time_index, hi_values)
    t_now = current_time if current_time is not None else float(np.asarray(time_index)[-1])
    return predict_time_to_threshold(a, b, threshold, t_now)
