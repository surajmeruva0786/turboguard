import numpy as np
from src.data.cwru_loader import align_channels, load_cwru_dataset

SYNTHETIC_DIR = "data/raw/cwru/synthetic"


def test_load_cwru_synthetic_dataset_shapes_and_labels():
    records = load_cwru_dataset(SYNTHETIC_DIR, source="synthetic")
    assert len(records) == 20  # 5 health states x 4 loads x 1 trial
    health_states = {r.health_state for r in records}
    assert health_states == {"healthy", "inner_race", "outer_race", "ball", "compound"}
    for r in records:
        assert r.signal.shape == (3, 12000)
        assert r.sample_rate_hz == 12000
        assert r.shaft_freq_hz > 0


def test_load_cwru_synthetic_dataset_known_shaft_speeds():
    records = load_cwru_dataset(SYNTHETIC_DIR, source="synthetic")
    rpms = sorted({r.rpm for r in records})
    assert rpms == [1730.0, 1750.0, 1772.0, 1797.0]


def test_align_channels_pads_when_too_few():
    signal = np.ones((1, 100))
    out = align_channels(signal, ("DE",), n_channels=3)
    assert out.shape == (3, 100)
    assert np.array_equal(out[0], out[1]) and np.array_equal(out[1], out[2])


def test_align_channels_truncates_when_too_many():
    signal = np.arange(4 * 10).reshape(4, 10)
    out = align_channels(signal, ("a", "b", "c", "d"), n_channels=3)
    assert out.shape == (3, 10)
    assert np.array_equal(out, signal[:3])


def test_align_channels_noop_when_exact():
    signal = np.ones((3, 10))
    out = align_channels(signal, ("a", "b", "c"), n_channels=3)
    assert out is signal
