import numpy as np
from src.preprocessing.conditioning import (
    antialias_lowpass,
    condition_signal,
    highpass_filter,
    remove_dc,
    resample_signal,
)
from src.preprocessing.windowing import window_count, window_signal


def test_remove_dc_zeros_mean():
    rng = np.random.default_rng(0)
    signal = rng.normal(5.0, 1.0, (3, 1000))
    out = remove_dc(signal)
    assert np.allclose(out.mean(axis=-1), 0, atol=1e-8)


def test_highpass_removes_slow_drift():
    sr = 1000.0
    t = np.arange(sr * 5) / sr
    drift = 3.0 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz, well below 5 Hz cutoff
    tone = 0.5 * np.sin(2 * np.pi * 50 * t)  # 50 Hz, should survive
    signal = (drift + tone)[None, :]
    out = highpass_filter(signal, sr, cutoff_hz=5.0)
    # drift should be strongly attenuated; overall variance should shrink a lot
    assert out.std() < signal.std()
    assert out.std() > 0.1  # but the 50 Hz tone should still be there


def test_antialias_lowpass_removes_high_frequency():
    sr = 12000.0
    t = np.arange(sr * 1) / sr
    low = np.sin(2 * np.pi * 500 * t)
    high = np.sin(2 * np.pi * 5900 * t)
    signal = (low + high)[None, :]
    out = antialias_lowpass(signal, sr, cutoff_hz=1000.0)
    # power should now be dominated by the low-frequency component
    freqs = np.fft.rfftfreq(len(t), 1 / sr)
    mag = np.abs(np.fft.rfft(out[0]))
    dominant_freq = freqs[np.argmax(mag)]
    assert abs(dominant_freq - 500) < 5


def test_resample_signal_changes_length_correctly():
    sr_orig, sr_target = 48000.0, 12000.0
    signal = np.zeros((2, 4800))
    out = resample_signal(signal, sr_orig, sr_target)
    assert out.shape[0] == 2
    assert out.shape[1] == 1200  # 4800 * (12000/48000)


def test_resample_signal_noop_when_rates_equal():
    signal = np.random.rand(2, 100)
    out = resample_signal(signal, 12000, 12000)
    assert np.array_equal(out, signal)


def test_condition_signal_runs_end_to_end():
    sr = 48000.0
    t = np.arange(sr * 1) / sr
    signal = (np.sin(2 * np.pi * 30 * t) + 0.01)[None, :]
    out = condition_signal(signal, orig_rate=sr, target_rate=12000.0)
    assert out.shape[1] == 12000
    assert abs(out.mean()) < 0.05


def test_window_signal_shapes_and_overlap():
    signal = np.arange(3 * 1000).reshape(3, 1000).astype(float)
    windows = window_signal(signal, sample_rate=1000, window_seconds=0.1, overlap=0.5)
    # window_samples=100, step=50 -> n_windows = (1000-100)//50 + 1 = 19
    assert windows.shape == (19, 3, 100)
    assert window_count(1000, 1000, 0.1, 0.5) == 19


def test_window_signal_no_overlap():
    signal = np.zeros((1, 300))
    windows = window_signal(signal, sample_rate=100, window_seconds=1.0, overlap=0.0)
    assert windows.shape == (3, 1, 100)


def test_window_signal_consecutive_no_cross_leakage():
    signal = np.arange(200).reshape(1, 200).astype(float)
    windows = window_signal(signal, sample_rate=100, window_seconds=1.0, overlap=0.5)
    # window 0 = samples [0,100), window 1 = samples [50,150)
    assert np.array_equal(windows[0, 0], signal[0, 0:100])
    assert np.array_equal(windows[1, 0], signal[0, 50:150])


def test_window_signal_too_short_returns_empty():
    signal = np.zeros((1, 10))
    windows = window_signal(signal, sample_rate=100, window_seconds=1.0, overlap=0.5)
    assert windows.shape == (0, 1, 100)
