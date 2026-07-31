import numpy as np
import pandas as pd

from src.preprocessing.run import process_cwru, process_ims
from pathlib import Path


def test_process_cwru_writes_windows_and_metadata(tmp_path):
    process_cwru(
        input_dir=Path("data/raw/cwru"),
        output_dir=tmp_path,
        source="synthetic",
        sfreq=12000.0,
        window_s=1.0,
        overlap=0.5,
        n_channels=3,
    )
    data = np.load(tmp_path / "windows.npz")
    assert data["X"].shape == (20, 3, 12000)
    meta = pd.read_csv(tmp_path / "metadata.csv")
    assert len(meta) == 20
    assert set(meta["health_state"]) == {"healthy", "inner_race", "outer_race", "ball", "compound"}


def test_process_ims_writes_windows_and_metadata(tmp_path):
    process_ims(
        input_dir=Path("data/raw/ims"),
        output_dir=tmp_path,
        source="synthetic",
        sfreq=12000.0,
        window_s=1.0,
        overlap=0.5,
        n_channels=3,
    )
    data = np.load(tmp_path / "windows.npz")
    assert data["X"].shape == (40, 3, 12000)
    meta = pd.read_csv(tmp_path / "metadata.csv")
    assert len(meta) == 40
    assert meta["rul_cycles"].min() == 0


def test_process_cwru_resamples_when_sfreq_differs(tmp_path):
    process_cwru(
        input_dir=Path("data/raw/cwru"),
        output_dir=tmp_path,
        source="synthetic",
        sfreq=6000.0,
        window_s=1.0,
        overlap=0.5,
        n_channels=3,
    )
    data = np.load(tmp_path / "windows.npz")
    assert data["X"].shape == (20, 3, 6000)
