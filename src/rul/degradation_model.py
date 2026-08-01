"""Exponential degradation model fit to a Health-Indicator trajectory (README 12.2, stage 2).

``HI(t) = a * exp(b * t)``, extrapolated to a configurable failure
threshold to produce a RUL estimate.
"""
from __future__ import annotations

import math

import numpy as np


def fit_exponential_degradation(
    time_index: np.ndarray, hi_values: np.ndarray, floor: float = 1e-6, max_growth_rate: float = 5.0
) -> tuple[float, float]:
    """Fit ``a, b`` in ``HI(t) = a * exp(b*t)`` via log-linear least squares.

    Linearising to ``log(HI) = log(a) + b*t`` and fitting with
    :func:`numpy.polyfit` is far more numerically stable on the handful of
    noisy points typical of an in-progress trajectory than nonlinear
    least-squares, which can diverge to absurd ``b`` values with few points
    (verified: unconstrained ``scipy.optimize.curve_fit`` produced
    ``b > 50`` and thresholds of ~1e21 on this project's tiny synthetic
    IMS set). ``hi_values`` are floored before the log to handle exact
    zeros; the fitted growth rate is clamped to ``max_growth_rate`` for the
    same reason.
    """
    time_index = np.asarray(time_index, dtype=np.float64)
    hi_safe = np.clip(np.asarray(hi_values, dtype=np.float64), floor, None)

    if len(time_index) < 2:
        return max(float(hi_safe[0]), floor), 0.0

    b, log_a = np.polyfit(time_index, np.log(hi_safe), deg=1)
    b = float(np.clip(b, -max_growth_rate, max_growth_rate))
    a = max(float(np.exp(log_a)), floor)
    return a, b


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
