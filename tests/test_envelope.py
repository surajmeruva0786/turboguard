import numpy as np
import pandas as pd
from src.features.bearing_freqs import CWRU_DRIVE_END_BEARING, bpfo
from src.features.envelope import dominant_envelope_peak, envelope_spectrum_features


def _load_synthetic_channel(path: str, channel: str = "acc_x") -> np.ndarray:
    df = pd.read_csv(path)
    return df[channel].to_numpy()


def test_dominant_envelope_peak_matches_bpfo_on_real_synthetic_data():
    x = _load_synthetic_channel("data/raw/cwru/synthetic/outer_race_load0_trial0.csv.gz")
    shaft_freq = 1797 / 60.0
    expected_bpfo = bpfo(shaft_freq, CWRU_DRIVE_END_BEARING)

    freq, _amp = dominant_envelope_peak(x, sample_rate=12000.0)
    assert abs(freq - expected_bpfo) < 2.0  # within 2 Hz, matching README's worked example


def test_envelope_features_bpfo_amplitude_higher_for_outer_race_than_healthy():
    x_outer = _load_synthetic_channel("data/raw/cwru/synthetic/outer_race_load0_trial0.csv.gz")
    x_healthy = _load_synthetic_channel("data/raw/cwru/synthetic/healthy_load0_trial0.csv.gz")
    shaft_freq = 1797 / 60.0

    f_outer = envelope_spectrum_features(x_outer, 12000.0, shaft_freq, CWRU_DRIVE_END_BEARING)
    f_healthy = envelope_spectrum_features(x_healthy, 12000.0, shaft_freq, CWRU_DRIVE_END_BEARING)

    assert f_outer["env_BPFO_h1_amp"] > f_healthy["env_BPFO_h1_amp"] * 3


def test_envelope_feature_keys_cover_all_characteristic_freqs_and_harmonics():
    x = np.random.default_rng(0).normal(0, 1, 12000)
    f = envelope_spectrum_features(x, 12000.0, 29.95, CWRU_DRIVE_END_BEARING, n_harmonics=3)
    for label in ("BPFO", "BPFI", "BSF", "FTF"):
        for h in (1, 2, 3):
            assert f"env_{label}_h{h}_amp" in f
    assert all(v >= 0 for v in f.values())
