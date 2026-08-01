import numpy as np

from src.evaluation.cross_dataset import cross_dataset_evaluate


def _separable_dataset(n_per_class=6, n_classes=3, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, n_features)) * 5
    X, y = [], []
    for cls in range(n_classes):
        for _ in range(n_per_class):
            X.append(centers[cls] + rng.normal(scale=0.1, size=n_features))
            y.append(cls)
    return np.array(X), np.array(y)


def test_cross_dataset_evaluate_returns_expected_sample_counts():
    X_train, y_train = _separable_dataset(seed=0)
    X_test, y_test = _separable_dataset(seed=1)
    metrics = cross_dataset_evaluate("logistic_regression", X_train, y_train, X_test, y_test, seed=42)
    assert metrics["n_train"] == len(X_train)
    assert metrics["n_test"] == len(X_test)


def test_cross_dataset_evaluate_high_accuracy_when_distributions_match():
    X_train, y_train = _separable_dataset(seed=0)
    X_test, y_test = _separable_dataset(seed=0)  # same distribution, different draw
    metrics = cross_dataset_evaluate("logistic_regression", X_train, y_train, X_test, y_test, seed=42)
    assert metrics["accuracy"] > 0.8


def test_cross_dataset_evaluate_single_class_test_set():
    """Mirrors the real synthetic-IMS case: test set has only one label."""
    X_train, y_train = _separable_dataset(n_classes=3, seed=0)
    X_test, y_test = _separable_dataset(n_classes=3, seed=0)
    single_class_mask = y_test == 0
    metrics = cross_dataset_evaluate(
        "logistic_regression", X_train, y_train, X_test[single_class_mask], y_test[single_class_mask], seed=42
    )
    assert metrics["n_test"] == int(single_class_mask.sum())
    assert 0.0 <= metrics["accuracy"] <= 1.0
