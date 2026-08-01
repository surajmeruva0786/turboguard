import torch
from src.training.augmentation import (
    add_gaussian_noise,
    amplitude_scale,
    augment_batch,
    time_shift,
)


def test_time_shift_preserves_shape_and_values():
    x = torch.randn(4, 3, 100)
    out = time_shift(x, max_shift_pct=0.1)
    assert out.shape == x.shape
    # a circular shift is a permutation, so the multiset of values is unchanged
    assert torch.allclose(out.sort(dim=-1).values, x.sort(dim=-1).values)


def test_amplitude_scale_preserves_shape():
    x = torch.randn(4, 3, 100)
    out = amplitude_scale(x, std=0.05)
    assert out.shape == x.shape


def test_amplitude_scale_is_per_sample_linear_scaling_of_zero_signal():
    x = torch.zeros(4, 3, 100)
    out = amplitude_scale(x, std=0.05)
    assert torch.allclose(out, x)


def test_add_gaussian_noise_increases_variance():
    torch.manual_seed(0)
    x = torch.ones(4, 3, 1000)
    out = add_gaussian_noise(x, snr_db=10.0)
    assert out.shape == x.shape
    assert out.var() > 0


def test_augment_batch_preserves_shape():
    x = torch.randn(4, 3, 12000)
    out = augment_batch(x)
    assert out.shape == x.shape


def test_augment_batch_works_on_sequence_input():
    x = torch.randn(2, 10, 3, 12000)
    out = augment_batch(x)
    assert out.shape == x.shape
