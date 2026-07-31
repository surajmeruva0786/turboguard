"""Combine per-axis time/frequency/envelope features + one shared wavelet
packet decomposition into the final ~180-dimensional feature vector
(README section 8.5) used by the classical baselines and as the
interpretable feature view for SHAP.

Wavelet packet features are computed once (on the first axis) rather than
per-axis, which is what keeps the final dimensionality around ~180 instead
of ~240 — per-axis time+frequency+envelope features (~48 each) for 3 axes
plus one shared 32-dim wavelet block: 3*48 + 32 = 176.
"""
from __future__ import annotations

import numpy as np

from src.features.bearing_freqs import BearingGeometry
from src.features.envelope import envelope_spectrum_features
from src.features.frequency_domain import frequency_domain_features
from src.features.time_domain import time_domain_features
from src.features.wavelet import wavelet_packet_features

DEFAULT_AXIS_NAMES = ("x", "y", "z")


def extract_feature_vector(
    window: np.ndarray,
    sample_rate: float,
    shaft_freq_hz: float | None = None,
    geom: BearingGeometry | None = None,
    axis_names: tuple[str, ...] = DEFAULT_AXIS_NAMES,
) -> dict[str, float]:
    """Extract the full named feature vector for one multi-channel window.

    ``window`` has shape ``(n_channels, n_samples)``. If ``shaft_freq_hz``
    and ``geom`` are omitted, physics-informed band-energy and envelope
    features are skipped (useful for exploratory analysis without known
    bearing geometry).
    """
    n_channels = window.shape[0]
    names = axis_names[:n_channels] if n_channels <= len(axis_names) else tuple(
        list(axis_names) + [f"ch{i}" for i in range(len(axis_names), n_channels)]
    )

    features: dict[str, float] = {}
    for i, axis in enumerate(names):
        x = window[i]
        for key, val in time_domain_features(x).items():
            features[f"{axis}_{key}"] = val
        for key, val in frequency_domain_features(x, sample_rate, shaft_freq_hz, geom).items():
            features[f"{axis}_{key}"] = val
        if shaft_freq_hz is not None and geom is not None:
            for key, val in envelope_spectrum_features(x, sample_rate, shaft_freq_hz, geom).items():
                features[f"{axis}_{key}"] = val

    for key, val in wavelet_packet_features(window[0]).items():
        features[key] = val

    return features


def feature_dict_to_array(feature_dicts: list[dict[str, float]]) -> tuple[np.ndarray, list[str]]:
    """Convert a list of per-window feature dicts (as returned repeatedly by
    :func:`extract_feature_vector`) into a ``(n_windows, n_features)`` array
    with a stable, sorted column order."""
    names = sorted(feature_dicts[0].keys())
    X = np.array([[d[name] for name in names] for d in feature_dicts], dtype=np.float64)
    return X, names
