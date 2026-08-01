#!/usr/bin/env python
"""Cross-condition evaluation (README 11.1): train a classical fault
classifier on a subset of CWRU load conditions, evaluate on a held-out
load — measuring generalisation to an unseen operating regime, as opposed
to the within-condition k-fold CV done in ``src.training.train_classical``.

Usage:
    python -m src.evaluation.cross_condition \\
        --model random_forest \\
        --dataset cwru \\
        --train_loads 0 1 2 \\
        --test_load 3 \\
        --output_dir results/cross_condition
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import FAULT_CLASSES
from src.evaluation.classification import classification_metrics
from src.features.extract import CWRU_METADATA_COLUMNS
from src.models.classical import ClassicalFaultClassifier
from src.utils.io import ensure_dir, save_json, save_yaml
from src.utils.logging_config import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def load_features_with_load(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_parquet(processed_dir / "features.parquet")
    feature_cols = [c for c in df.columns if c not in CWRU_METADATA_COLUMNS]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["health_state"].map(FAULT_CLASSES.index).to_numpy()
    loads = df["load_hp"].to_numpy()
    return X, y, loads, feature_cols


def cross_condition_evaluate(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    loads: np.ndarray,
    train_loads: list[int],
    test_load: int,
    seed: int = 42,
    **model_kwargs,
) -> dict:
    """Fit on samples whose load is in ``train_loads``, evaluate on ``test_load``."""
    train_mask = np.isin(loads, train_loads)
    test_mask = loads == test_load
    if not test_mask.any():
        raise ValueError(f"No samples found for test_load={test_load}")
    if not train_mask.any():
        raise ValueError(f"No samples found for train_loads={train_loads}")

    clf = ClassicalFaultClassifier(model_name, seed=seed, **model_kwargs)
    clf.fit(X[train_mask], y[train_mask])
    y_pred = clf.predict(X[test_mask])
    y_proba = clf.predict_proba(X[test_mask])

    proba_full = np.zeros((test_mask.sum(), len(FAULT_CLASSES)))
    for i, cls in enumerate(clf.classes_):
        proba_full[:, cls] = y_proba[:, i]

    metrics = classification_metrics(y[test_mask], y_pred, proba_full, class_names=FAULT_CLASSES)
    metrics["n_train"] = int(train_mask.sum())
    metrics["n_test"] = int(test_mask.sum())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--dataset", choices=["cwru"], default="cwru")
    parser.add_argument("--processed_dir", type=Path, default=None)
    parser.add_argument("--train_loads", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--test_load", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    processed_dir = args.processed_dir or Path("data/processed") / args.dataset
    X, y, loads, feature_cols = load_features_with_load(processed_dir)
    logger.info("Loaded %d samples x %d features from %s", *X.shape, processed_dir)

    metrics = cross_condition_evaluate(args.model, X, y, loads, args.train_loads, args.test_load, args.seed)
    logger.info(
        "Cross-condition (train=%s -> test=%s): accuracy=%.3f macro_f1=%.3f",
        args.train_loads,
        args.test_load,
        metrics["accuracy"],
        metrics["macro_f1"],
    )

    output_dir = ensure_dir(args.output_dir)
    save_json(metrics, output_dir / "metrics.json")
    save_yaml(
        {
            "model": args.model,
            "dataset": args.dataset,
            "train_loads": args.train_loads,
            "test_load": args.test_load,
            "seed": args.seed,
            "n_features": len(feature_cols),
        },
        output_dir / "config.yaml",
    )
    logger.info("Saved cross-condition metrics + config -> %s", output_dir)


if __name__ == "__main__":
    main()
