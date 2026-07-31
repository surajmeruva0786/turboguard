"""Classical ML baselines for fault classification (README section 9.1).

All four models share one interface (:class:`ClassicalFaultClassifier`) so
the training CLI, evaluation, and SHAP explainer don't need to special-case
any particular model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODEL_FACTORIES = {
    "random_forest": lambda seed, **kw: RandomForestClassifier(
        n_estimators=kw.get("n_estimators", 500),
        max_depth=kw.get("max_depth"),
        random_state=seed,
        n_jobs=-1,
    ),
    "xgboost": lambda seed, **kw: XGBClassifier(
        n_estimators=kw.get("n_estimators", 1000),
        learning_rate=kw.get("learning_rate", 0.05),
        max_depth=kw.get("max_depth", 6),
        random_state=seed,
        eval_metric="mlogloss",
    ),
    "svm": lambda seed, **kw: SVC(
        kernel="rbf", C=kw.get("C", 1.0), gamma=kw.get("gamma", "scale"), probability=True, random_state=seed
    ),
    "logistic_regression": lambda seed, **kw: LogisticRegression(
        penalty="l2", C=kw.get("C", 1.0), max_iter=kw.get("max_iter", 1000), random_state=seed
    ),
}


class ClassicalFaultClassifier:
    """A classical model + its feature scaler, bundled for consistent
    fit/predict/save/load regardless of which sklearn/xgboost estimator
    backs it."""

    def __init__(self, model_name: str, seed: int = 42, scale_features: bool = True, **model_kwargs: Any):
        if model_name not in MODEL_FACTORIES:
            raise ValueError(f"Unknown model_name {model_name!r}; choose from {list(MODEL_FACTORIES)}")
        self.model_name = model_name
        self.model = MODEL_FACTORIES[model_name](seed, **model_kwargs)
        self.scaler = StandardScaler() if scale_features else None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalFaultClassifier":
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.classes_ = np.asarray(self.model.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "model_name": self.model_name, "classes_": self.classes_},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ClassicalFaultClassifier":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.model_name = data["model_name"]
        obj.classes_ = data["classes_"]
        return obj
