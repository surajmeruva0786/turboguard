import numpy as np
from src.features.wavelet import wavelet_packet_features


def test_produces_16_nodes_for_level_4():
    x = np.random.default_rng(0).normal(0, 1, 4096)
    f = wavelet_packet_features(x, level=4)
    assert len(f) == 32  # 16 nodes x (rel_energy, entropy)
    rel_energy_keys = [k for k in f if k.endswith("rel_energy")]
    assert len(rel_energy_keys) == 16


def test_relative_energies_sum_to_approximately_one():
    x = np.random.default_rng(1).normal(0, 1, 4096)
    f = wavelet_packet_features(x, level=4)
    total = sum(v for k, v in f.items() if k.endswith("rel_energy"))
    assert abs(total - 1.0) < 1e-6


def test_all_values_finite_and_nonnegative():
    x = np.random.default_rng(2).normal(0, 1, 4096)
    f = wavelet_packet_features(x, level=4)
    assert all(np.isfinite(v) for v in f.values())
    assert all(v >= 0 for k, v in f.items() if k.endswith("rel_energy"))


def test_constant_signal_concentrates_energy_in_lowest_node():
    x = np.full(4096, 2.0)
    f = wavelet_packet_features(x, level=4)
    rel_energies = [f[f"wp_node{i}_rel_energy"] for i in range(16)]
    assert np.argmax(rel_energies) == 0  # DC content lives in the lowest-frequency node
    assert rel_energies[0] > 0.9
