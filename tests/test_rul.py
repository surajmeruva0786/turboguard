import numpy as np
import pytest
import torch

from src.rul.combine import combine_rul_estimates, learn_combination_weight
from src.rul.degradation_model import (
    estimate_rul_from_trajectory,
    fit_exponential_degradation,
    predict_time_to_threshold,
)
from src.rul.direct_regression import piecewise_linear_rul_target, predict_rul
from src.rul.health_indicator import fit_health_indicator_model
from src.models.turboguard_hybrid import TurboGuardHybrid


def test_piecewise_linear_rul_target_caps_at_default():
    assert piecewise_linear_rul_target(200.0) == 125.0
    assert piecewise_linear_rul_target(50.0) == 50.0


def test_piecewise_linear_rul_target_respects_custom_cap():
    assert piecewise_linear_rul_target(200.0, cap=50.0) == 50.0


def test_predict_rul_returns_expected_shape():
    model = TurboGuardHybrid()
    windows = torch.randn(3, 10, 3, 12000)
    preds = predict_rul(model, windows)
    assert preds.shape == (3,)


def test_fit_health_indicator_model_reduces_reconstruction_error():
    rng = np.random.default_rng(0)
    X_healthy = rng.normal(size=(20, 10))
    hi_model = fit_health_indicator_model(X_healthy, latent_dim=3, hidden_dims=(6,), epochs=100)
    hi_healthy = hi_model.compute(X_healthy)
    assert hi_healthy.shape == (20,)
    assert np.all(hi_healthy >= 0)


def test_health_indicator_model_flags_out_of_distribution_samples_higher():
    rng = np.random.default_rng(1)
    X_healthy = rng.normal(loc=0.0, scale=1.0, size=(30, 8))
    hi_model = fit_health_indicator_model(X_healthy, latent_dim=3, hidden_dims=(6,), epochs=300)
    hi_healthy = hi_model.compute(X_healthy)

    X_degraded = rng.normal(loc=10.0, scale=1.0, size=(10, 8))
    hi_degraded = hi_model.compute(X_degraded)

    assert hi_degraded.mean() > hi_healthy.mean()


def test_fit_exponential_degradation_recovers_known_parameters():
    t = np.arange(20, dtype=np.float64)
    a_true, b_true = 1.0, 0.2
    hi = a_true * np.exp(b_true * t)
    a_fit, b_fit = fit_exponential_degradation(t, hi)
    assert a_fit == pytest.approx(a_true, rel=0.05)
    assert b_fit == pytest.approx(b_true, rel=0.05)


def test_predict_time_to_threshold_positive_growth():
    rul = predict_time_to_threshold(a=1.0, b=0.2, threshold=10.0, current_time=5.0)
    assert rul > 0


def test_predict_time_to_threshold_nan_for_non_increasing_trajectory():
    assert np.isnan(predict_time_to_threshold(a=1.0, b=-0.1, threshold=10.0))


def test_estimate_rul_from_trajectory_decreases_as_degradation_progresses():
    t = np.arange(20, dtype=np.float64)
    hi = 0.5 * np.exp(0.15 * t)
    threshold = float(hi[-1]) * 2

    rul_early = estimate_rul_from_trajectory(t[:5], hi[:5], threshold)
    rul_late = estimate_rul_from_trajectory(t[:15], hi[:15], threshold, current_time=14.0)
    assert rul_late < rul_early


def test_combine_rul_estimates_weighted_average():
    combined = combine_rul_estimates(direct_pred=10.0, hi_pred=20.0, weight_direct=0.25)
    assert combined == pytest.approx(0.25 * 10.0 + 0.75 * 20.0)


def test_learn_combination_weight_prefers_more_accurate_estimator():
    actual = np.array([10.0, 20.0, 30.0])
    direct_preds = actual.copy()  # perfect
    hi_preds = actual + 50.0  # way off
    w = learn_combination_weight(direct_preds, hi_preds, actual)
    assert w > 0.9
