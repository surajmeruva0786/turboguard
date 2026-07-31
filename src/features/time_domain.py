"""Time-domain statistical features (README section 8.1), computed per axis.

Crest factor and kurtosis in particular rise sharply as impulsive bearing
defects develop, making them strong early-fault indicators even before any
frequency-domain analysis.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew

_EPS = 1e-12


def _hjorth_params(x: np.ndarray) -> tuple[float, float, float]:
    """Hjorth activity, mobility, complexity."""
    d1 = np.diff(x)
    d2 = np.diff(d1)
    var_x = np.var(x)
    var_d1 = np.var(d1)
    var_d2 = np.var(d2)

    activity = var_x
    mobility = np.sqrt(var_d1 / (var_x + _EPS))
    mobility_d1 = np.sqrt(var_d2 / (var_d1 + _EPS))
    complexity = mobility_d1 / (mobility + _EPS)
    return float(activity), float(mobility), float(complexity)


def time_domain_features(x: np.ndarray) -> dict[str, float]:
    """Compute the full time-domain feature set for a single-axis 1D signal."""
    x = np.asarray(x, dtype=np.float64)
    abs_x = np.abs(x)
    mean_abs = abs_x.mean()
    rms = np.sqrt(np.mean(x**2))
    peak = abs_x.max() if x.size else 0.0
    sqrt_abs_mean = np.mean(np.sqrt(abs_x))

    activity, mobility, complexity = _hjorth_params(x)

    signs = np.sign(x)
    signs[signs == 0] = 1
    zero_crossings = np.sum(np.diff(signs) != 0)

    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "var": float(x.var()),
        "rms": float(rms),
        "peak": float(peak),
        "peak_to_peak": float(x.max() - x.min()),
        "crest_factor": float(peak / (rms + _EPS)),
        "shape_factor": float(rms / (mean_abs + _EPS)),
        "impulse_factor": float(peak / (mean_abs + _EPS)),
        "margin_factor": float(peak / (sqrt_abs_mean**2 + _EPS)),
        "skewness": float(_skew(x)),
        "kurtosis": float(_kurtosis(x)),  # Fisher (excess) kurtosis
        "line_length": float(np.sum(np.abs(np.diff(x)))),
        "zero_crossing_rate": float(zero_crossings / len(x)),
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }
