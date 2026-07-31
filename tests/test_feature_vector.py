import numpy as np
import pandas as pd

from src.features.bearing_freqs import CWRU_DRIVE_END_BEARING
from src.features.feature_vector import extract_feature_vector, feature_dict_to_array


def _load_synthetic_window(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    return df[["acc_x", "acc_y", "acc_z"]].to_numpy().T


def test_feature_vector_dimensionality_near_180():
    window = _load_synthetic_window("data/raw/cwru/synthetic/healthy_load0_trial0.csv.gz")
    feats = extract_feature_vector(window, 12000.0, 1797 / 60.0, CWRU_DRIVE_END_BEARING)
    assert 150 <= len(feats) <= 200


def test_feature_vector_without_geom_skips_physics_features():
    window = _load_synthetic_window("data/raw/cwru/synthetic/healthy_load0_trial0.csv.gz")
    feats = extract_feature_vector(window, 12000.0)
    assert not any(k.endswith("band_energy_BPFO") for k in feats)
    assert not any("env_BPFO" in k for k in feats)


def test_feature_vector_all_finite():
    window = _load_synthetic_window("data/raw/cwru/synthetic/inner_race_load1_trial0.csv.gz")
    feats = extract_feature_vector(window, 12000.0, 1772 / 60.0, CWRU_DRIVE_END_BEARING)
    assert all(np.isfinite(v) for v in feats.values())


def test_outer_race_has_higher_bpfo_band_energy_than_healthy():
    shaft_freq = 1797 / 60.0
    outer = _load_synthetic_window("data/raw/cwru/synthetic/outer_race_load0_trial0.csv.gz")
    healthy = _load_synthetic_window("data/raw/cwru/synthetic/healthy_load0_trial0.csv.gz")

    f_outer = extract_feature_vector(outer, 12000.0, shaft_freq, CWRU_DRIVE_END_BEARING)
    f_healthy = extract_feature_vector(healthy, 12000.0, shaft_freq, CWRU_DRIVE_END_BEARING)

    assert f_outer["x_band_energy_BPFO"] > f_healthy["x_band_energy_BPFO"]
    assert f_outer["x_env_BPFO_h1_amp"] > f_healthy["x_env_BPFO_h1_amp"]


def test_feature_dict_to_array_consistent_ordering():
    window = _load_synthetic_window("data/raw/cwru/synthetic/ball_load0_trial0.csv.gz")
    feats1 = extract_feature_vector(window, 12000.0, 1797 / 60.0, CWRU_DRIVE_END_BEARING)
    feats2 = extract_feature_vector(window, 12000.0, 1797 / 60.0, CWRU_DRIVE_END_BEARING)
    X, names = feature_dict_to_array([feats1, feats2])
    assert X.shape == (2, len(names))
    assert np.allclose(X[0], X[1])  # deterministic given the same input
    assert names == sorted(names)
