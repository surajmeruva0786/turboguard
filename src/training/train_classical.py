#!/usr/bin/env python
"""Train + cross-validate a classical fault-classification baseline.

Usage (README section 16):
    python -m src.training.train_classical \\
        --model xgboost \\
        --dataset cwru \\
        --output_dir runs/xgb_cwru

Requires ``data/processed/<dataset>/features.parquet`` (produced by
``src.preprocessing.run`` + ``src.features.extract``). Saves the model
fit on all data plus out-of-fold cross-validated metrics to
``--output_dir``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.dataset import FAULT_CLASSES
from src.evaluation.classification import classification_metrics
from src.features.extract import CWRU_METADATA_COLUMNS
from src.models.classical import MODEL_FACTORIES, ClassicalFaultClassifier
from src.utils.io import ensure_dir, save_json, save_yaml
from src.utils.logging_config import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def load_cwru_features(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_parquet(processed_dir / "features.parquet")
    feature_cols = [c for c in df.columns if c not in CWRU_METADATA_COLUMNS]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["health_state"].map(FAULT_CLASSES.index).to_numpy()
    return X, y, feature_cols


def cross_validated_fit(
    model_name: str, X: np.ndarray, y: np.ndarray, cv_folds: int, seed: int, **model_kwargs
) -> tuple[ClassicalFaultClassifier, dict]:
    """Out-of-fold CV metrics, then a final model refit on all data."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_pred = np.zeros_like(y)
    oof_proba = np.zeros((len(y), len(FAULT_CLASSES)))

    for train_idx, test_idx in skf.split(X, y):
        fold_clf = ClassicalFaultClassifier(model_name, seed=seed, **model_kwargs)
        fold_clf.fit(X[train_idx], y[train_idx])
        oof_pred[test_idx] = fold_clf.predict(X[test_idx])
        proba = fold_clf.predict_proba(X[test_idx])
        for i, cls in enumerate(fold_clf.classes_):
            oof_proba[test_idx, cls] = proba[:, i]

    metrics = classification_metrics(y, oof_pred, oof_proba, class_names=FAULT_CLASSES)

    final_clf = ClassicalFaultClassifier(model_name, seed=seed, **model_kwargs)
    final_clf.fit(X, y)
    return final_clf, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODEL_FACTORIES), required=True)
    parser.add_argument("--dataset", choices=["cwru"], default="cwru")
    parser.add_argument("--processed_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--cv_folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    processed_dir = args.processed_dir or Path("data/processed") / args.dataset
    X, y, feature_cols = load_cwru_features(processed_dir)
    logger.info("Loaded %d samples x %d features from %s", *X.shape, processed_dir)

    clf, metrics = cross_validated_fit(args.model, X, y, args.cv_folds, args.seed)
    logger.info("CV accuracy=%.3f macro_f1=%.3f", metrics["accuracy"], metrics["macro_f1"])

    output_dir = ensure_dir(args.output_dir)
    clf.save(output_dir / "model.joblib")
    save_json(metrics, output_dir / "metrics.json")
    save_yaml(
        {"model": args.model, "dataset": args.dataset, "cv_folds": args.cv_folds, "seed": args.seed,
         "n_features": len(feature_cols)},
        output_dir / "config.yaml",
    )
    logger.info("Saved model + metrics + config -> %s", output_dir)


if __name__ == "__main__":
    main()
