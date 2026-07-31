"""Envelope-spectrum analysis (README section 8.3) — the classical bearing-
diagnostics method the deep model is benchmarked against.

The vibration signal is band-pass filtered around the structural resonance
excited by bearing impacts, the Hilbert envelope isolates the impact-train
amplitude modulation, and the envelope's own FFT reveals the bearing
characteristic frequencies as clean peaks (rather than being buried under
broadband high-frequency resonance content).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from src.features.bearing_freqs import BearingGeometry, all_characteristic_frequencies

_EPS = 1e-12


def envelope_spectrum(
    x: np.ndarray, sample_rate: float, band: tuple[float, float] = (2000.0, 5900.0), order: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Band-pass -> Hilbert envelope -> FFT. Returns ``(freqs, magnitude)``."""
    x = np.asarray(x, dtype=np.float64)
    low, high = band
    nyquist = sample_rate / 2
    high = min(high, nyquist * 0.98)
    sos = butter(order, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    filtered = sosfiltfilt(sos, x)

    envelope = np.abs(hilbert(filtered))
    envelope = envelope - envelope.mean()

    freqs = np.fft.rfftfreq(len(envelope), 1 / sample_rate)
    mag = np.abs(np.fft.rfft(envelope))
    return freqs, mag


def dominant_envelope_peak(
    x: np.ndarray, sample_rate: float, band: tuple[float, float] = (2000.0, 5900.0)
) -> tuple[float, float]:
    """Return ``(frequency_hz, amplitude)`` of the single largest envelope-spectrum
    peak (excluding DC) — used by the XAI bearing-frequency annotator."""
    freqs, mag = envelope_spectrum(x, sample_rate, band)
    idx = int(np.argmax(mag[1:])) + 1 if len(mag) > 1 else 0
    return float(freqs[idx]), float(mag[idx])


def _peak_amplitude_near(freqs: np.ndarray, mag: np.ndarray, target_hz: float, tol_hz: float) -> float:
    mask = (freqs >= target_hz - tol_hz) & (freqs <= target_hz + tol_hz)
    if not mask.any():
        return 0.0
    return float(mag[mask].max())


def envelope_spectrum_features(
    x: np.ndarray,
    sample_rate: float,
    shaft_freq_hz: float,
    geom: BearingGeometry,
    band: tuple[float, float] = (2000.0, 5900.0),
    n_harmonics: int = 3,
    tol_hz: float = 2.0,
) -> dict[str, float]:
    """Peak envelope-spectrum amplitude at each characteristic frequency and
    its first ``n_harmonics`` harmonics."""
    freqs, mag = envelope_spectrum(x, sample_rate, band)
    char_freqs = all_characteristic_frequencies(shaft_freq_hz, geom)

    features: dict[str, float] = {}
    for label, base_freq in char_freqs.items():
        for h in range(1, n_harmonics + 1):
            features[f"env_{label}_h{h}_amp"] = _peak_amplitude_near(freqs, mag, base_freq * h, tol_hz)
    return features
