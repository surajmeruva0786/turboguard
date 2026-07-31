"""Loaders for the CWRU Bearing Dataset (real .mat files or the synthetic stand-in).

Real CWRU files are single- or dual-axis (drive-end / fan-end / occasionally
base accelerometer), unlike the synthetic generator's fixed triaxial output.
:func:`align_channels` reconciles the two so downstream code always sees a
fixed channel count.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_CHANNELS = ("acc_x", "acc_y", "acc_z")

# Real CWRU .mat variable naming, e.g. "X097_DE_time", "X097RPM".
_CHANNEL_VAR_RE = re.compile(r"X\d+_(DE|FE|BA)_time")
_RPM_VAR_RE = re.compile(r"X\d+RPM")

CWRU_LOAD_TO_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


@dataclass
class CWRURecord:
    signal: np.ndarray  # (n_channels, n_samples)
    channel_names: tuple[str, ...]
    health_state: str
    load_hp: int
    rpm: float
    shaft_freq_hz: float
    sample_rate_hz: float
    fault_diameter_in: float = 0.0
    source_file: str = ""
    extra: dict = field(default_factory=dict)


def align_channels(signal: np.ndarray, channel_names: tuple[str, ...], n_channels: int = 3) -> np.ndarray:
    """Pad or truncate ``signal`` to exactly ``n_channels`` rows.

    Real CWRU recordings often have only 1-2 axes available. Missing
    channels are filled by repeating the last available channel (better
    than zeros for feature extraction, which would otherwise see spuriously
    "dead" axes); extra channels beyond ``n_channels`` are dropped.
    """
    c = signal.shape[0]
    if c == n_channels:
        return signal
    if c > n_channels:
        return signal[:n_channels]
    pad = np.repeat(signal[-1:], n_channels - c, axis=0)
    return np.concatenate([signal, pad], axis=0)


def load_cwru_synthetic_dataset(input_dir: str | Path) -> list[CWRURecord]:
    """Load the synthetic CWRU-like dataset produced by ``scripts/generate_synthetic.py``."""
    input_dir = Path(input_dir)
    labels = pd.read_csv(input_dir / "labels.csv")
    records = []
    for row in labels.itertuples():
        df = pd.read_csv(input_dir / row.filename)
        signal = df[list(CANONICAL_CHANNELS)].to_numpy().T
        records.append(
            CWRURecord(
                signal=signal,
                channel_names=CANONICAL_CHANNELS,
                health_state=row.health_state,
                load_hp=int(row.load_hp),
                rpm=float(row.rpm),
                shaft_freq_hz=float(row.shaft_freq_hz),
                sample_rate_hz=float(row.sample_rate_hz),
                fault_diameter_in=0.0,
                source_file=row.filename,
            )
        )
    return records


def load_cwru_mat_file(path: str | Path) -> dict[str, np.ndarray]:
    """Load one real CWRU ``.mat`` file, returning ``{"DE": arr, "FE": arr, ...}``
    plus ``"RPM"`` if present. Requires ``scipy``."""
    from scipy.io import loadmat

    mat = loadmat(str(path))
    out: dict[str, np.ndarray] = {}
    for key, value in mat.items():
        m = _CHANNEL_VAR_RE.match(key)
        if m:
            out[m.group(1)] = np.asarray(value).squeeze().astype(np.float64)
        elif _RPM_VAR_RE.match(key):
            out["RPM"] = np.asarray(value).squeeze().astype(np.float64)
    return out


def load_cwru_real_dataset(input_dir: str | Path) -> list[CWRURecord]:
    """Load real CWRU data given a ``manifest.csv`` produced by
    ``scripts/download_cwru.py`` (columns: category, label, fault_location,
    fault_diameter_in, load_hp, sample_rate_hz, filename)."""
    input_dir = Path(input_dir)
    manifest = pd.read_csv(input_dir / "manifest.csv")
    records = []
    for row in manifest.itertuples():
        mat_path = input_dir / row.category / row.filename
        channels = load_cwru_mat_file(mat_path)
        rpm_arr = channels.pop("RPM", None)
        rpm = float(rpm_arr.flat[0]) if rpm_arr is not None and rpm_arr.size else CWRU_LOAD_TO_RPM[row.load_hp]
        channel_names = tuple(sorted(channels))
        if not channel_names:
            continue
        signal = np.stack([channels[c] for c in channel_names])
        records.append(
            CWRURecord(
                signal=signal,
                channel_names=channel_names,
                health_state=row.fault_location,
                load_hp=int(row.load_hp),
                rpm=rpm,
                shaft_freq_hz=rpm / 60.0,
                sample_rate_hz=float(row.sample_rate_hz),
                fault_diameter_in=float(row.fault_diameter_in),
                source_file=row.filename,
            )
        )
    return records


def load_cwru_dataset(input_dir: str | Path, source: str = "synthetic") -> list[CWRURecord]:
    if source == "synthetic":
        return load_cwru_synthetic_dataset(input_dir)
    if source == "real":
        return load_cwru_real_dataset(input_dir)
    raise ValueError(f"Unknown source: {source!r} (expected 'synthetic' or 'real')")
