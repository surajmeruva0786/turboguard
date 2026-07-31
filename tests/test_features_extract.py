from pathlib import Path

from src.features.extract import extract_cwru, extract_ims
from src.preprocessing.run import process_cwru, process_ims


def test_extract_cwru_end_to_end(tmp_path):
    process_cwru(Path("data/raw/cwru"), tmp_path, "synthetic", 12000.0, 1.0, 0.5, 3)
    df = extract_cwru(tmp_path, tmp_path, 12000.0)
    assert len(df) == 20
    assert "health_state" in df.columns
    assert "x_kurtosis" in df.columns
    assert df.filter(like="_amp").shape[1] > 0
    assert (tmp_path / "features.parquet").exists()


def test_extract_ims_end_to_end(tmp_path):
    process_ims(Path("data/raw/ims"), tmp_path, "synthetic", 12000.0, 1.0, 0.5, 3)
    df = extract_ims(tmp_path, tmp_path, 12000.0)
    assert len(df) == 40
    assert "rul_cycles" in df.columns
    assert "x_band_energy_BPFO" in df.columns


def test_extract_cwru_outer_race_rows_have_higher_bpfo_energy(tmp_path):
    process_cwru(Path("data/raw/cwru"), tmp_path, "synthetic", 12000.0, 1.0, 0.5, 3)
    df = extract_cwru(tmp_path, tmp_path, 12000.0)
    outer = df[df["health_state"] == "outer_race"]["x_band_energy_BPFO"]
    healthy = df[df["health_state"] == "healthy"]["x_band_energy_BPFO"]
    assert outer.mean() > healthy.mean()
