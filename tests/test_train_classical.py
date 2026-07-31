from pathlib import Path

from src.features.extract import extract_cwru
from src.preprocessing.run import process_cwru
from src.training.train_classical import cross_validated_fit, load_cwru_features


def _prepare_features(tmp_path):
    process_cwru(Path("data/raw/cwru"), tmp_path, "synthetic", 12000.0, 1.0, 0.5, 3)
    extract_cwru(tmp_path, tmp_path, 12000.0)
    return tmp_path


def test_load_cwru_features_shapes(tmp_path):
    processed_dir = _prepare_features(tmp_path)
    X, y, feature_cols = load_cwru_features(processed_dir)
    assert X.shape == (20, len(feature_cols))
    assert y.shape == (20,)
    assert set(y) == {0, 1, 2, 3, 4}


def test_cross_validated_fit_returns_metrics_and_fitted_model(tmp_path):
    processed_dir = _prepare_features(tmp_path)
    X, y, _ = load_cwru_features(processed_dir)
    clf, metrics = cross_validated_fit("random_forest", X, y, cv_folds=4, seed=42)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "confusion_matrix" in metrics
    preds = clf.predict(X)
    assert preds.shape == y.shape


def test_easily_separable_synthetic_faults_achieve_high_cv_accuracy(tmp_path):
    # Sanity check that the pipeline (features -> classifier -> CV) actually
    # learns something on this clearly-separable synthetic data, rather than
    # e.g. silently training on all-zero features.
    processed_dir = _prepare_features(tmp_path)
    X, y, _ = load_cwru_features(processed_dir)
    _, metrics = cross_validated_fit("xgboost", X, y, cv_folds=4, seed=42)
    assert metrics["accuracy"] >= 0.8
