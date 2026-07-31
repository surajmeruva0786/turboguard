import numpy as np

from src.features.bearing_freqs import CWRU_DRIVE_END_BEARING, bpfo
from src.features.frequency_domain import frequency_domain_features


def test_single_tone_mean_frequency_matches():
    sr = 12000.0
    t = np.arange(sr * 1) / sr
    x = np.sin(2 * np.pi * 500 * t)
    f = frequency_domain_features(x, sr)
    assert abs(f["mean_frequency_hz"] - 500) < 10
    assert abs(f["median_frequency_hz"] - 500) < 10


def test_top_fft_peak_matches_tone_frequency_and_is_largest():
    sr = 12000.0
    t = np.arange(sr * 1) / sr
    x = 2.0 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.sin(2 * np.pi * 1000 * t)
    f = frequency_domain_features(x, sr)
    assert abs(f["fft_peak1_freq_hz"] - 300) < 2
    assert f["fft_peak1_amp"] > f["fft_peak2_amp"]


def test_band_energy_higher_when_signal_has_bpfo_content():
    sr = 12000.0
    shaft_freq = 29.95
    t = np.arange(sr * 1) / sr
    bpfo_freq = bpfo(shaft_freq, CWRU_DRIVE_END_BEARING)

    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 0.05, len(t))
    with_bpfo = baseline + 1.0 * np.sin(2 * np.pi * bpfo_freq * t)

    f_base = frequency_domain_features(baseline, sr, shaft_freq, CWRU_DRIVE_END_BEARING)
    f_bpfo = frequency_domain_features(with_bpfo, sr, shaft_freq, CWRU_DRIVE_END_BEARING)

    assert f_bpfo["band_energy_BPFO"] > f_base["band_energy_BPFO"] * 10


def test_all_expected_keys_present_with_geom():
    sr = 12000.0
    x = np.random.default_rng(1).normal(0, 1, 4000)
    f = frequency_domain_features(x, sr, 29.95, CWRU_DRIVE_END_BEARING)
    for label in ("BPFO", "BPFI", "BSF", "FTF"):
        assert f"band_energy_{label}" in f
    for i in range(1, 6):
        assert f"fft_peak{i}_freq_hz" in f
        assert f"fft_peak{i}_amp" in f


def test_features_finite():
    x = np.random.default_rng(2).normal(0, 1, 3000)
    f = frequency_domain_features(x, 12000.0)
    assert all(np.isfinite(v) for v in f.values())
