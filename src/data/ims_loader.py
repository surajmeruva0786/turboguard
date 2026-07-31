"""Loaders for the NASA IMS Bearing Dataset (real snapshot files or the synthetic stand-in)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_CHANNELS = ("acc_x", "acc_y", "acc_z")

# Real IMS test sets record all bearings' channels in one whitespace-delimited
# snapshot file per timestamp. Test 1 has 2 channels/bearing (8 columns total);
# tests 2 and 3 have 1 channel/bearing (4 columns total). Source: Qiu et al.
# (2006); verify against the README shipped inside the downloaded archive
# before relying on this for anything beyond a quick look, as we do not have
# it open here to double check byte-for-byte.
IMS_TEST_SET_LAYOUT = {
    1: {"n_bearings": 4, "channels_per_bearing": 2},
    2: {"n_bearings": 4, "channels_per_bearing": 1},
    3: {"n_bearings": 4, "channels_per_bearing": 1},
}


@dataclass
class IMSRecord:
    signal: np.ndarray  # (n_channels, n_samples)
    bearing_id: int
    snapshot_index: int
    rul_cycles: int
    health_indicator: float
    sample_rate_hz: float
    dominant_fault: str = ""
    source_file: str = ""


def load_ims_synthetic_dataset(input_dir: str | Path, bearing_id: int | None = None) -> list[IMSRecord]:
    """Load the synthetic IMS-like run-to-failure dataset. If ``bearing_id`` is
    given, only that bearing's trajectory is returned (already sorted by
    ``snapshot_index``); otherwise all bearings are returned concatenated."""
    input_dir = Path(input_dir)
    labels = pd.read_csv(input_dir / "labels.csv")
    if bearing_id is not None:
        labels = labels[labels["bearing_id"] == bearing_id]
    labels = labels.sort_values(["bearing_id", "snapshot_index"])

    records = []
    for row in labels.itertuples():
        df = pd.read_csv(input_dir / row.filename)
        signal = df[list(CANONICAL_CHANNELS)].to_numpy().T
        records.append(
            IMSRecord(
                signal=signal,
                bearing_id=int(row.bearing_id),
                snapshot_index=int(row.snapshot_index),
                rul_cycles=int(row.rul_cycles),
                health_indicator=float(row.health_indicator),
                sample_rate_hz=float(row.sample_rate_hz),
                dominant_fault=row.dominant_fault,
                source_file=row.filename,
            )
        )
    return records


def _parse_snapshot_timestamp(filename: str) -> str:
    # Real IMS filenames are literal timestamps, e.g. "2003.10.22.12.06.24".
    return filename


def load_ims_real_test_set(test_dir: str | Path, test_set: int, bearing_id: int) -> list[IMSRecord]:
    """Load one bearing's real run-to-failure trajectory from an extracted IMS
    test-set directory (e.g. ``data/raw/ims/1st_test``).

    Snapshot files have no header and no reliable per-row timestamp; ``rul_cycles``
    is therefore a snapshot-count proxy (snapshots-until-last-file), not a
    calibrated time-to-failure — consistent with how the direct-regression
    target is defined in :mod:`src.rul.direct_regression`.
    """
    test_dir = Path(test_dir)
    layout = IMS_TEST_SET_LAYOUT[test_set]
    cpb = layout["channels_per_bearing"]
    col_start = (bearing_id - 1) * cpb
    col_end = col_start + cpb

    files = sorted(p for p in test_dir.iterdir() if p.is_file())
    n_total = len(files)
    records = []
    for idx, path in enumerate(files):
        arr = np.loadtxt(path, delimiter="\t")
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        channels = arr[:, col_start:col_end].T
        records.append(
            IMSRecord(
                signal=channels,
                bearing_id=bearing_id,
                snapshot_index=idx,
                rul_cycles=n_total - 1 - idx,
                health_indicator=float("nan"),
                sample_rate_hz=20000.0,
                dominant_fault="unknown",
                source_file=_parse_snapshot_timestamp(path.name),
            )
        )
    return records


def load_ims_dataset(input_dir: str | Path, source: str = "synthetic", **kwargs) -> list[IMSRecord]:
    if source == "synthetic":
        return load_ims_synthetic_dataset(input_dir, bearing_id=kwargs.get("bearing_id"))
    if source == "real":
        return load_ims_real_test_set(input_dir, kwargs["test_set"], kwargs["bearing_id"])
    raise ValueError(f"Unknown source: {source!r} (expected 'synthetic' or 'real')")
