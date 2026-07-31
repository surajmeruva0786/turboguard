"""Frequency-domain features (README section 8.2), computed per axis.

Includes the physics-informed bearing characteristic-frequency band
energies (BPFO/BPFI/BSF/FTF), which is what makes this feature set
directly comparable to classical envelope-analysis diagnostics.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, welch

from src.features.bearing_freqs import BearingGeometry, all_characteristic_frequencies

_EPS = 1e-12


def _psd_moments(freqs: np.ndarray, psd: np.ndarray) -> tuple[float, float, float, float]:
    """Mean, median, spread (std), and excess kurtosis of the PSD-as-distribution."""
    total_power = psd.sum() + _EPS
    mean_freq = float(np.sum(freqs * psd) / total_power)

    cumulative = np.cumsum(psd) / total_power
    median_idx = int(np.searchsorted(cumulative, 0.5))
    median_freq = float(freqs[min(median_idx, len(freqs) - 1)])

    spread = float(np.sqrt(np.sum(((freqs - mean_freq) ** 2) * psd) / total_power))
    fourth_moment = np.sum(((freqs - mean_freq) ** 4) * psd) / total_power
    spectral_kurtosis = float(fourth_moment / (spread**4 + _EPS) - 3.0)

    return mean_freq, median_freq, spread, spectral_kurtosis


def _spectral_centroid(freqs: np.ndarray, mag: np.ndarray) -> float:
    total = mag.sum() + _EPS
    return float(np.sum(freqs * mag) / total)


def _band_energy(freqs: np.ndarray, psd: np.ndarray, center_hz: float, halfwidth_hz: float) -> float:
    mask = (freqs >= center_hz - halfwidth_hz) & (freqs <= center_hz + halfwidth_hz)
    return float(psd[mask].sum())


def frequency_domain_features(
    x: np.ndarray,
    sample_rate: float,
    shaft_freq_hz: float | None = None,
    geom: BearingGeometry | None = None,
    band_halfwidth_hz: float = 5.0,
    n_peaks: int = 5,
) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    nperseg = min(4096, len(x))
    freqs, psd = welch(x, fs=sample_rate, nperseg=nperseg)

    mean_freq, median_freq, spread, spec_kurt = _psd_moments(freqs, psd)

    fft_freqs = np.fft.rfftfreq(len(x), 1 / sample_rate)
    fft_mag = np.abs(np.fft.rfft(x))
    centroid = _spectral_centroid(fft_freqs, fft_mag)

    features = {
        "mean_frequency_hz": mean_freq,
        "median_frequency_hz": median_freq,
        "spectral_centroid_hz": centroid,
        "spectral_spread_hz": spread,
        "spectral_kurtosis": spec_kurt,
    }

    if shaft_freq_hz is not None and geom is not None:
        char_freqs = all_characteristic_frequencies(shaft_freq_hz, geom)
        for label, freq_hz in char_freqs.items():
            features[f"band_energy_{label}"] = _band_energy(freqs, psd, freq_hz, band_halfwidth_hz)

    # Top-N FFT peaks by amplitude (excluding DC bin).
    peak_indices, _ = find_peaks(fft_mag[1:])
    peak_indices = peak_indices + 1
    if len(peak_indices) > 0:
        order = np.argsort(fft_mag[peak_indices])[::-1][:n_peaks]
        top_peaks = peak_indices[order]
    else:
        top_peaks = np.array([], dtype=int)

    for i in range(n_peaks):
        if i < len(top_peaks):
            idx = top_peaks[i]
            features[f"fft_peak{i + 1}_freq_hz"] = float(fft_freqs[idx])
            features[f"fft_peak{i + 1}_amp"] = float(fft_mag[idx])
        else:
            features[f"fft_peak{i + 1}_freq_hz"] = 0.0
            features[f"fft_peak{i + 1}_amp"] = 0.0

    return features
