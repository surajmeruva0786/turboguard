"""Fixed-length windowing with overlap (README section 7.2)."""
from __future__ import annotations

import numpy as np


def window_signal(
    signal: np.ndarray, sample_rate: float, window_seconds: float = 1.0, overlap: float = 0.5
) -> np.ndarray:
    """Slice ``signal`` (n_channels, n_samples) into overlapping fixed-length windows.

    Returns an array of shape ``(n_windows, n_channels, window_samples)``.
    Windows are only ever formed from *consecutive* samples within the same
    recording, so no cross-recording leakage is possible.
    """
    window_samples = int(window_seconds * sample_rate)
    if window_samples <= 0:
        raise ValueError("window_seconds * sample_rate must be >= 1 sample")
    step = max(1, int(window_samples * (1 - overlap)))

    n_channels, n_samples = signal.shape
    n_windows = max(0, (n_samples - window_samples) // step + 1)
    if n_windows == 0:
        return np.empty((0, n_channels, window_samples), dtype=signal.dtype)

    windows = np.empty((n_windows, n_channels, window_samples), dtype=signal.dtype)
    for i in range(n_windows):
        start = i * step
        windows[i] = signal[:, start : start + window_samples]
    return windows


def window_count(n_samples: int, sample_rate: float, window_seconds: float = 1.0, overlap: float = 0.5) -> int:
    window_samples = int(window_seconds * sample_rate)
    step = max(1, int(window_samples * (1 - overlap)))
    return max(0, (n_samples - window_samples) // step + 1)
