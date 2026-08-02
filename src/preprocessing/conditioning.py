"""Signal conditioning: DC removal, drift/anti-alias filtering, resampling.

See README section 7.1. Applied per-channel, before windowing.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


def remove_dc(signal: np.ndarray) -> np.ndarray:
    """Subtract the per-channel running mean (DC offset)."""
    return signal - signal.mean(axis=-1, keepdims=True)


def highpass_filter(signal: np.ndarray, sample_rate: float, cutoff_hz: float = 5.0, order: int = 4) -> np.ndarray:
    """Remove residual low-frequency drift below ``cutoff_hz``."""
    sos = butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, signal, axis=-1)


def antialias_lowpass(signal: np.ndarray, sample_rate: float, cutoff_hz: float = 6000.0, order: int = 8) -> np.ndarray:
    """Low-pass before resampling, to avoid aliasing."""
    nyquist = sample_rate / 2
    cutoff_hz = min(cutoff_hz, nyquist * 0.98)
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, signal, axis=-1)


def resample_signal(signal: np.ndarray, orig_rate: float, target_rate: float) -> np.ndarray:
    """Resample to ``target_rate`` using polyphase filtering (exact rational resampling)."""
    if orig_rate == target_rate:
        return signal
    from math import gcd

    g = gcd(int(orig_rate), int(target_rate))
    up, down = int(target_rate) // g, int(orig_rate) // g
    return resample_poly(signal, up, down, axis=-1)


def condition_signal(
    signal: np.ndarray,
    orig_rate: float,
    target_rate: float = 12000.0,
    highpass_cutoff_hz: float = 5.0,
    antialias_cutoff_hz: float = 6000.0,
) -> np.ndarray:
    """Full conditioning chain: DC removal -> high-pass -> anti-alias -> resample."""
    signal = remove_dc(signal)
    signal = highpass_filter(signal, orig_rate, highpass_cutoff_hz)
    if target_rate < orig_rate:
        signal = antialias_lowpass(signal, orig_rate, antialias_cutoff_hz)
    signal = resample_signal(signal, orig_rate, target_rate)
    return signal


def resample_and_fix_length(
    signal: np.ndarray, sample_rate: float, target_rate: float | None, window_seconds: float | None
) -> np.ndarray:
    """Resample to ``target_rate`` (if given) then pad/truncate to exactly
    ``window_seconds`` long. Lets recordings taken at a different native rate
    or duration (e.g. real IMS's 20 kHz/1.024 s snapshots vs CWRU's 12 kHz/1 s
    windows) come out as the same fixed shape a downstream fixed-input model
    expects. No-op when both are ``None``."""
    if target_rate and target_rate != sample_rate:
        signal = condition_signal(signal, sample_rate, target_rate)
        sample_rate = target_rate
    if not window_seconds:
        return signal
    n = int(round(window_seconds * sample_rate))
    if signal.shape[-1] >= n:
        return signal[:, :n]
    return np.pad(signal, ((0, 0), (0, n - signal.shape[-1])))
