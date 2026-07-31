#!/usr/bin/env python
"""Preprocessing CLI: raw records -> conditioned, windowed, saved arrays.

Usage (README section 16):
    python -m src.preprocessing.run \\
        --dataset cwru \\
        --input_dir data/raw/cwru \\
        --output_dir data/processed/cwru \\
        --sfreq 12000 \\
        --window 1.0 \\
        --overlap 0.5

Writes ``windows.npz`` (array ``X`` of shape ``(n_windows, n_channels,
window_samples)``) and ``metadata.csv`` (one row per window, in the same
order as ``X``) to ``--output_dir``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cwru_loader import align_channels, load_cwru_dataset
from src.data.ims_loader import load_ims_dataset
from src.preprocessing.conditioning import condition_signal
from src.preprocessing.windowing import window_signal
from src.utils.io import ensure_dir
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _resolve_input_dir(input_dir: Path, dataset: str, source: str) -> Path:
    """README examples pass the dataset's base dir (e.g. data/raw/cwru); for
    synthetic data the actual files live one level down, in ``synthetic/``."""
    if source == "synthetic" and input_dir.name != "synthetic":
        candidate = input_dir / "synthetic"
        if candidate.exists():
            return candidate
    return input_dir


def process_cwru(
    input_dir: Path, output_dir: Path, source: str, sfreq: float, window_s: float, overlap: float, n_channels: int
) -> None:
    records = load_cwru_dataset(_resolve_input_dir(input_dir, "cwru", source), source=source)
    all_windows, meta_rows = [], []
    for r in records:
        signal = r.signal
        if signal.shape[0] != n_channels:
            signal = align_channels(signal, r.channel_names, n_channels=n_channels)
        if r.sample_rate_hz != sfreq:
            signal = condition_signal(signal, r.sample_rate_hz, sfreq)
        windows = window_signal(signal, sfreq, window_s, overlap)
        for i, w in enumerate(windows):
            all_windows.append(w.astype(np.float32))
            meta_rows.append(
                {
                    "health_state": r.health_state,
                    "load_hp": r.load_hp,
                    "rpm": r.rpm,
                    "shaft_freq_hz": r.shaft_freq_hz,
                    "fault_diameter_in": r.fault_diameter_in,
                    "source_file": r.source_file,
                    "window_index": i,
                }
            )
    _write_outputs(all_windows, meta_rows, output_dir)


def process_ims(
    input_dir: Path, output_dir: Path, source: str, sfreq: float, window_s: float, overlap: float, n_channels: int
) -> None:
    records = load_ims_dataset(_resolve_input_dir(input_dir, "ims", source), source=source)
    all_windows, meta_rows = [], []
    for r in records:
        signal = r.signal
        if signal.shape[0] != n_channels:
            signal = align_channels(signal, (), n_channels=n_channels)
        if r.sample_rate_hz != sfreq:
            signal = condition_signal(signal, r.sample_rate_hz, sfreq)
        windows = window_signal(signal, sfreq, window_s, overlap)
        for i, w in enumerate(windows):
            all_windows.append(w.astype(np.float32))
            meta_rows.append(
                {
                    "bearing_id": r.bearing_id,
                    "snapshot_index": r.snapshot_index,
                    "rul_cycles": r.rul_cycles,
                    "health_indicator": r.health_indicator,
                    "dominant_fault": r.dominant_fault,
                    "source_file": r.source_file,
                    "window_index": i,
                }
            )
    _write_outputs(all_windows, meta_rows, output_dir)


def _write_outputs(windows: list[np.ndarray], meta_rows: list[dict], output_dir: Path) -> None:
    output_dir = ensure_dir(output_dir)
    X = np.stack(windows) if windows else np.empty((0, 0, 0), dtype=np.float32)
    np.savez_compressed(output_dir / "windows.npz", X=X)
    pd.DataFrame(meta_rows).to_csv(output_dir / "metadata.csv", index=False)
    logger.info("Wrote %d windows -> %s", len(windows), output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cwru", "ims"], required=True)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--sfreq", type=float, default=12000.0)
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--n_channels", type=int, default=3)
    args = parser.parse_args()

    if args.dataset == "cwru":
        process_cwru(args.input_dir, args.output_dir, args.source, args.sfreq, args.window, args.overlap, args.n_channels)
    else:
        process_ims(args.input_dir, args.output_dir, args.source, args.sfreq, args.window, args.overlap, args.n_channels)


if __name__ == "__main__":
    main()
