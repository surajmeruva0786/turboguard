import numpy as np
import pytest

from src.evaluation.cross_condition import cross_condition_evaluate


def _synthetic_load_dataset(n_per_class_per_load: int = 3, n_classes: int = 3, loads=(0, 1, 2, 3), seed=0):
    rng = np.random.default_rng(seed)
    X, y, load_arr = [], [], []
    centers = rng.normal(size=(n_classes, 5)) * 5
    for load in loads:
        for cls in range(n_classes):
            for _ in range(n_per_class_per_load):
                X.append(centers[cls] + rng.normal(scale=0.1, size=5))
                y.append(cls)
                load_arr.append(load)
    return np.array(X), np.array(y), np.array(load_arr)


def test_cross_condition_evaluate_trains_only_on_specified_loads():
    X, y, loads = _synthetic_load_dataset()
    metrics = cross_condition_evaluate(
        "logistic_regression", X, y, loads, train_loads=[0, 1, 2], test_load=3, seed=42
    )
    assert metrics["n_train"] == 3 * 3 * 3  # 3 loads x 3 classes x 3 samples
    assert metrics["n_test"] == 3 * 3  # 1 load x 3 classes x 3 samples


def test_cross_condition_evaluate_achieves_high_accuracy_on_separable_classes():
    X, y, loads = _synthetic_load_dataset()
    metrics = cross_condition_evaluate(
        "logistic_regression", X, y, loads, train_loads=[0, 1, 2], test_load=3, seed=42
    )
    assert metrics["accuracy"] > 0.8


def test_cross_condition_evaluate_raises_on_missing_test_load():
    X, y, loads = _synthetic_load_dataset()
    with pytest.raises(ValueError, match="test_load"):
        cross_condition_evaluate("logistic_regression", X, y, loads, train_loads=[0, 1], test_load=99)


def test_cross_condition_evaluate_raises_on_missing_train_loads():
    X, y, loads = _synthetic_load_dataset()
    with pytest.raises(ValueError, match="train_loads"):
        cross_condition_evaluate("logistic_regression", X, y, loads, train_loads=[99], test_load=0)
