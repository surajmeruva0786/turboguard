#!/usr/bin/env python
"""Feature extraction CLI: processed windows -> the ~180-dim feature vector.

Usage:
    python -m src.features.extract --dataset cwru --input_dir data/processed/cwru --output_dir data/processed/cwru
    python -m src.features.extract --dataset ims  --input_dir data/processed/ims  --output_dir data/processed/ims

Reads ``windows.npz``/``metadata.csv`` written by :mod:`src.preprocessing.run`
and writes ``features.parquet`` (metadata columns + all feature columns,
one row per window) to ``--output_dir``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.bearing_freqs import CWRU_DRIVE_END_BEARING, IMS_BEARING, IMS_SHAFT_FREQ_HZ
from src.features.feature_vector import extract_feature_vector, feature_dict_to_array
from src.utils.io import ensure_dir
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_cwru(input_dir: Path, output_dir: Path, sample_rate: float) -> pd.DataFrame:
    X = np.load(input_dir / "windows.npz")["X"]
    meta = pd.read_csv(input_dir / "metadata.csv")
    feature_dicts = [
        extract_feature_vector(X[i], sample_rate, meta.iloc[i]["shaft_freq_hz"], CWRU_DRIVE_END_BEARING)
        for i in range(len(X))
    ]
    return _assemble(meta, feature_dicts, output_dir)


def extract_ims(input_dir: Path, output_dir: Path, sample_rate: float) -> pd.DataFrame:
    X = np.load(input_dir / "windows.npz")["X"]
    meta = pd.read_csv(input_dir / "metadata.csv")
    feature_dicts = [
        extract_feature_vector(X[i], sample_rate, IMS_SHAFT_FREQ_HZ, IMS_BEARING) for i in range(len(X))
    ]
    return _assemble(meta, feature_dicts, output_dir)


def _assemble(meta: pd.DataFrame, feature_dicts: list[dict[str, float]], output_dir: Path) -> pd.DataFrame:
    Xf, names = feature_dict_to_array(feature_dicts)
    features_df = pd.DataFrame(Xf, columns=names)
    combined = pd.concat([meta.reset_index(drop=True), features_df], axis=1)
    output_dir = ensure_dir(output_dir)
    combined.to_parquet(output_dir / "features.parquet", index=False)
    logger.info("Wrote %d rows x %d feature columns -> %s", len(combined), len(names), output_dir / "features.parquet")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cwru", "ims"], required=True)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sfreq", type=float, default=12000.0)
    args = parser.parse_args()

    if args.dataset == "cwru":
        extract_cwru(args.input_dir, args.output_dir, args.sfreq)
    else:
        extract_ims(args.input_dir, args.output_dir, args.sfreq)


if __name__ == "__main__":
    main()
