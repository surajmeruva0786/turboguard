"""Wavelet packet decomposition features (README section 8.4).

A 4-level db4 wavelet packet decomposition splits the signal into 16
equal-width frequency sub-bands; the relative energy and Shannon entropy of
each sub-band capture how the signal's energy is distributed across
frequency in a way that complements the single-point FFT/PSD features.
"""
from __future__ import annotations

import numpy as np
import pywt

_EPS = 1e-12


def wavelet_packet_features(x: np.ndarray, wavelet: str = "db4", level: int = 4) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    wp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode="symmetric", maxlevel=level)
    nodes = wp.get_level(level, order="natural")

    energies = [float(np.sum(node.data**2)) for node in nodes]
    total_energy = sum(energies) + _EPS

    features: dict[str, float] = {}
    for i, (node, energy) in enumerate(zip(nodes, energies)):
        rel_energy = energy / total_energy
        coeffs = node.data
        p = (coeffs**2) / (energy + _EPS)
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log2(p))) if len(p) else 0.0
        features[f"wp_node{i}_rel_energy"] = rel_energy
        features[f"wp_node{i}_entropy"] = entropy
    return features
