import numpy as np

from src.features.time_domain import time_domain_features


def test_constant_signal_features():
    x = np.full(1000, 3.0)
    f = time_domain_features(x)
    assert f["mean"] == 3.0
    assert f["std"] == 0.0
    assert f["peak_to_peak"] == 0.0
    assert f["zero_crossing_rate"] == 0.0


def test_sine_wave_rms_and_crest_factor():
    t = np.linspace(0, 1, 12000, endpoint=False)
    x = np.sin(2 * np.pi * 50 * t)
    f = time_domain_features(x)
    assert abs(f["rms"] - 1 / np.sqrt(2)) < 0.01
    assert abs(f["crest_factor"] - np.sqrt(2)) < 0.05
    assert abs(f["peak"] - 1.0) < 0.01


def test_impulsive_signal_has_higher_crest_factor_and_kurtosis():
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 5000)
    impulsive = baseline.copy()
    impulsive[::200] += 15.0  # sparse large spikes -> impulsive

    f_base = time_domain_features(baseline)
    f_imp = time_domain_features(impulsive)
    assert f_imp["crest_factor"] > f_base["crest_factor"]
    assert f_imp["kurtosis"] > f_base["kurtosis"]


def test_zero_crossing_rate_of_alternating_signal():
    x = np.array([1.0, -1.0] * 500)
    f = time_domain_features(x)
    # every consecutive pair flips sign -> 999 crossings / 1000 samples
    assert abs(f["zero_crossing_rate"] - 0.999) < 1e-9


def test_all_features_present_and_finite():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 2000)
    f = time_domain_features(x)
    expected_keys = {
        "mean", "std", "var", "rms", "peak", "peak_to_peak", "crest_factor",
        "shape_factor", "impulse_factor", "margin_factor", "skewness",
        "kurtosis", "line_length", "zero_crossing_rate", "hjorth_activity",
        "hjorth_mobility", "hjorth_complexity",
    }
    assert set(f) == expected_keys
    assert all(np.isfinite(v) for v in f.values())
