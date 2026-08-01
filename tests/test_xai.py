import numpy as np
import pytest
from src.features.bearing_freqs import CWRU_DRIVE_END_BEARING
from src.models.classical import ClassicalFaultClassifier
from src.xai.bearing_freq_annotator import annotate_alert
from src.xai.shap_deep import explain_deep_model, train_feature_mlp
from src.xai.shap_tree import (
    explain_tree_model,
    top_features_for_class,
    top_features_for_prediction,
)


def _toy_classification_data(n_per_class=8, n_classes=3, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, n_features)) * 5
    X, y = [], []
    for cls in range(n_classes):
        for _ in range(n_per_class):
            X.append(centers[cls] + rng.normal(scale=0.1, size=n_features))
            y.append(cls)
    return np.array(X), np.array(y)


def test_explain_tree_model_shape():
    X, y = _toy_classification_data()
    clf = ClassicalFaultClassifier("random_forest", seed=42, n_estimators=20)
    clf.fit(X, y)
    shap_values = explain_tree_model(clf, X)
    assert shap_values.shape == (len(X), X.shape[1], 3)


def test_explain_tree_model_raises_for_non_tree_model():
    X, y = _toy_classification_data()
    clf = ClassicalFaultClassifier("logistic_regression", seed=42)
    clf.fit(X, y)
    with pytest.raises(ValueError, match="TreeExplainer"):
        explain_tree_model(clf, X)


def test_top_features_for_prediction_returns_top_n_sorted_by_magnitude():
    X, y = _toy_classification_data()
    clf = ClassicalFaultClassifier("random_forest", seed=42, n_estimators=20)
    clf.fit(X, y)
    shap_values = explain_tree_model(clf, X)
    top = top_features_for_prediction(shap_values, [f"f{i}" for i in range(X.shape[1])], 0, 0, top_n=3)
    assert len(top) == 3
    magnitudes = [abs(v) for _, v in top]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_top_features_for_class_returns_top_n():
    X, y = _toy_classification_data()
    clf = ClassicalFaultClassifier("random_forest", seed=42, n_estimators=20)
    clf.fit(X, y)
    shap_values = explain_tree_model(clf, X)
    top = top_features_for_class(shap_values, [f"f{i}" for i in range(X.shape[1])], class_idx=0, top_n=4)
    assert len(top) == 4


def test_train_feature_mlp_and_explain_deep_model_shape():
    X, y = _toy_classification_data()
    model = train_feature_mlp(X, y, epochs=50)
    shap_values = explain_deep_model(model, X, X[:5])
    assert shap_values.shape == (5, X.shape[1], 3)


def test_annotate_alert_produces_readable_message():
    rng = np.random.default_rng(0)
    x = rng.normal(size=12000)
    annotation = annotate_alert(
        x,
        sample_rate=12000.0,
        shaft_freq_hz=29.95,
        geom=CWRU_DRIVE_END_BEARING,
        predicted_class="outer_race",
        confidence=0.94,
    )
    assert "Outer-race fault" in annotation.message
    assert "Confidence 0.94" in annotation.message
    assert "Hz matches" in annotation.message
