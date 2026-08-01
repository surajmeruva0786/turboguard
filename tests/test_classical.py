import numpy as np
import pytest
from sklearn.datasets import make_classification
from src.models.classical import ClassicalFaultClassifier

MODEL_NAMES = ["random_forest", "xgboost", "svm", "logistic_regression"]


def _toy_dataset():
    X, y = make_classification(
        n_samples=200, n_features=20, n_informative=10, n_classes=5, n_clusters_per_class=1, random_state=42
    )
    return X, y


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_fit_predict_shapes(model_name):
    X, y = _toy_dataset()
    clf = ClassicalFaultClassifier(model_name, seed=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    proba = clf.predict_proba(X)
    assert preds.shape == y.shape
    assert proba.shape == (len(y), 5)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_reasonable_accuracy_on_easy_toy_data(model_name):
    X, y = _toy_dataset()
    clf = ClassicalFaultClassifier(model_name, seed=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    accuracy = (preds == y).mean()
    assert accuracy > 0.5  # well above chance (1/5) on training data itself


def test_save_load_roundtrip_preserves_predictions(tmp_path):
    X, y = _toy_dataset()
    clf = ClassicalFaultClassifier("random_forest", seed=42)
    clf.fit(X, y)
    preds_before = clf.predict(X)

    path = tmp_path / "model.joblib"
    clf.save(path)
    loaded = ClassicalFaultClassifier.load(path)
    preds_after = loaded.predict(X)

    assert np.array_equal(preds_before, preds_after)


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        ClassicalFaultClassifier("not_a_real_model")
