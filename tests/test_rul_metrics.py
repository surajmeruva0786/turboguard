import numpy as np
import pytest

from src.evaluation.rul_metrics import mape, phm_score, rmse, rul_metrics


def test_rmse_zero_for_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0])
    assert rmse(y, y) == 0.0


def test_rmse_matches_hand_computation():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert rmse(y_true, y_pred) == pytest.approx(3.5355339059327378)


def test_mape_zero_for_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0])
    assert mape(y, y) == 0.0


def test_mape_handles_zero_true_value_without_error():
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([1.0, 11.0])
    result = mape(y_true, y_pred)
    assert np.isfinite(result)


def test_phm_score_zero_for_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0])
    assert phm_score(y, y) == 0.0


def test_phm_score_penalises_late_predictions_more_than_early():
    y_true = np.array([10.0])
    early_pred = np.array([5.0])  # d = -5 (early)
    late_pred = np.array([15.0])  # d = +5 (late)
    assert phm_score(y_true, late_pred) > phm_score(y_true, early_pred)


def test_rul_metrics_returns_all_keys():
    y_true = np.array([10.0, 5.0, 0.0])
    y_pred = np.array([12.0, 4.0, 1.0])
    result = rul_metrics(y_true, y_pred)
    assert set(result.keys()) == {"rmse", "mape", "phm_score", "n_samples"}
    assert result["n_samples"] == 3
