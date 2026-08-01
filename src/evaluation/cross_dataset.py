#!/usr/bin/env python
"""Cross-dataset evaluation (README 11.1): train a classical fault
classifier on CWRU, evaluate on IMS — the toughest generalisation test,
across both equipment and sensor setup.

Usage:
    python -m src.evaluation.cross_dataset \\
        --model random_forest \\
        --train_dataset cwru \\
        --test_dataset ims \\
        --output_dir results/cross_dataset

Note: the synthetic IMS set's every snapshot is labelled ``outer_race``
(a run-to-failure trajectory has one dominant fault mode, unlike CWRU's
per-class trials), so this is necessarily a single-class evaluation on
the IMS side — a real accuracy/recall check, not a full confusion matrix.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import FAULT_CLASSES
from src.evaluation.classification import classification_metrics
from src.features.extract import CWRU_METADATA_COLUMNS, IMS_METADATA_COLUMNS
from src.models.classical import ClassicalFaultClassifier
from src.utils.io import ensure_dir, save_json, save_yaml
from src.utils.logging_config import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

LABEL_COLUMNS = {"cwru": "health_state", "ims": "dominant_fault"}
METADATA_COLUMNS = {"cwru": CWRU_METADATA_COLUMNS, "ims": IMS_METADATA_COLUMNS}


def load_labelled_features(dataset: str, processed_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_parquet(processed_dir / "features.parquet")
    feature_cols = [c for c in df.columns if c not in METADATA_COLUMNS[dataset]]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[LABEL_COLUMNS[dataset]].map(FAULT_CLASSES.index).to_numpy()
    return X, y, feature_cols


def cross_dataset_evaluate(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    **model_kwargs,
) -> dict:
    clf = ClassicalFaultClassifier(model_name, seed=seed, **model_kwargs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    proba_full = np.zeros((len(X_test), len(FAULT_CLASSES)))
    for i, cls in enumerate(clf.classes_):
        proba_full[:, cls] = y_proba[:, i]

    metrics = classification_metrics(y_test, y_pred, proba_full, class_names=FAULT_CLASSES)
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--train_dataset", choices=["cwru", "ims"], default="cwru")
    parser.add_argument("--test_dataset", choices=["cwru", "ims"], default="ims")
    parser.add_argument("--train_processed_dir", type=Path, default=None)
    parser.add_argument("--test_processed_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    train_dir = args.train_processed_dir or Path("data/processed") / args.train_dataset
    test_dir = args.test_processed_dir or Path("data/processed") / args.test_dataset

    X_train, y_train, train_cols = load_labelled_features(args.train_dataset, train_dir)
    X_test, y_test, test_cols = load_labelled_features(args.test_dataset, test_dir)
    common_cols = [c for c in train_cols if c in set(test_cols)]
    if len(common_cols) != len(train_cols) or len(common_cols) != len(test_cols):
        logger.warning(
            "Feature columns differ between %s (%d) and %s (%d); using %d common columns",
            args.train_dataset, len(train_cols), args.test_dataset, len(test_cols), len(common_cols),
        )
        train_idx = [train_cols.index(c) for c in common_cols]
        test_idx = [test_cols.index(c) for c in common_cols]
        X_train, X_test = X_train[:, train_idx], X_test[:, test_idx]

    logger.info("Train: %d samples from %s. Test: %d samples from %s.", len(X_train), args.train_dataset, len(X_test), args.test_dataset)
    metrics = cross_dataset_evaluate(args.model, X_train, y_train, X_test, y_test, args.seed)
    logger.info(
        "Cross-dataset (%s -> %s): accuracy=%.3f macro_f1=%.3f",
        args.train_dataset, args.test_dataset, metrics["accuracy"], metrics["macro_f1"],
    )

    output_dir = ensure_dir(args.output_dir)
    save_json(metrics, output_dir / "metrics.json")
    save_yaml(
        {
            "model": args.model,
            "train_dataset": args.train_dataset,
            "test_dataset": args.test_dataset,
            "seed": args.seed,
            "n_common_features": len(common_cols),
        },
        output_dir / "config.yaml",
    )
    logger.info("Saved cross-dataset metrics + config -> %s", output_dir)


if __name__ == "__main__":
    main()
