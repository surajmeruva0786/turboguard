"""SHAP TreeExplainer for the classical Random Forest / XGBoost baselines (README 13.1).

Maps raw SHAP attributions back to the named engineered features (e.g.
"BPFI band energy", "crest factor on axis y") so the operator sees a
human-readable reason for a prediction.
"""
from __future__ import annotations

import numpy as np
import shap

from src.models.classical import ClassicalFaultClassifier

TREE_MODELS = {"random_forest", "xgboost"}


def explain_tree_model(clf: ClassicalFaultClassifier, X: np.ndarray) -> np.ndarray:
    """SHAP values for a fitted tree-based :class:`ClassicalFaultClassifier`.

    Returns an array of shape ``(n_samples, n_features, n_classes)``.
    Raises ``ValueError`` for non-tree models (SVM/logistic regression need
    a different explainer, e.g. ``shap.KernelExplainer``, out of scope here).
    """
    if clf.model_name not in TREE_MODELS:
        raise ValueError(f"TreeExplainer requires one of {TREE_MODELS}, got {clf.model_name!r}")

    X_input = clf.scaler.transform(X) if clf.scaler is not None else X
    explainer = shap.TreeExplainer(clf.model)
    raw = explainer.shap_values(X_input)

    if isinstance(raw, list):  # older SHAP API: one (n_samples, n_features) array per class
        return np.stack(raw, axis=-1)
    return raw  # newer SHAP API: already (n_samples, n_features, n_classes)


def top_features_for_prediction(
    shap_values: np.ndarray,
    feature_names: list[str],
    sample_idx: int,
    class_idx: int,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """The ``top_n`` named features with the largest |SHAP value| for one
    sample's attribution to one class, sorted most-influential first."""
    values = shap_values[sample_idx, :, class_idx]
    order = np.argsort(-np.abs(values))[:top_n]
    return [(feature_names[i], float(values[i])) for i in order]


def top_features_for_class(
    shap_values: np.ndarray, feature_names: list[str], class_idx: int, top_n: int = 10
) -> list[tuple[str, float]]:
    """The ``top_n`` named features with the largest mean |SHAP value| across
    all samples for one class — a global, per-class importance ranking."""
    mean_abs = np.abs(shap_values[:, :, class_idx]).mean(axis=0)
    order = np.argsort(-mean_abs)[:top_n]
    return [(feature_names[i], float(mean_abs[i])) for i in order]
