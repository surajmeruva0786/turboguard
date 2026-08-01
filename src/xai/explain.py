#!/usr/bin/env python
"""Generate a SHAP + physical-justification explanation report for one
prediction (README section 13, roadmap step 89).

Usage:
    python -m src.xai.explain \\
        --model_dir runs/random_forest_cwru \\
        --processed_dir data/processed/cwru \\
        --sample_idx 0 \\
        --output_dir results/xai/sample_0

Loads a trained :class:`ClassicalFaultClassifier`, explains one sample with
``shap.TreeExplainer`` (top-10 contributing named features), and annotates
it with the closest bearing characteristic frequency (README 13.3).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import FAULT_CLASSES
from src.features.bearing_freqs import CWRU_DRIVE_END_BEARING
from src.features.extract import CWRU_METADATA_COLUMNS
from src.models.classical import ClassicalFaultClassifier
from src.xai.bearing_freq_annotator import annotate_alert
from src.xai.shap_tree import explain_tree_model, top_features_for_prediction
from src.utils.io import ensure_dir, save_json
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_sample_explanation(
    model_dir: Path, processed_dir: Path, sample_idx: int, top_n: int = 10
) -> dict:
    clf = ClassicalFaultClassifier.load(model_dir / "model.joblib")
    df = pd.read_parquet(processed_dir / "features.parquet")
    feature_cols = [c for c in df.columns if c not in CWRU_METADATA_COLUMNS]
    X = df[feature_cols].to_numpy(dtype=np.float64)

    row = df.iloc[sample_idx]
    pred = clf.predict(X[[sample_idx]])[0]
    proba = clf.predict_proba(X[[sample_idx]])[0]
    pred_class = FAULT_CLASSES[pred]
    confidence = float(proba[list(clf.classes_).index(pred)])

    shap_values = explain_tree_model(clf, X[[sample_idx]])
    top_features = top_features_for_prediction(shap_values, feature_cols, 0, pred, top_n)

    windows = np.load(processed_dir / "windows.npz")["X"]
    window = windows[sample_idx]
    annotation = annotate_alert(
        window[0],
        sample_rate=12000.0,
        shaft_freq_hz=float(row["shaft_freq_hz"]),
        geom=CWRU_DRIVE_END_BEARING,
        predicted_class=pred_class,
        confidence=confidence,
    )

    return {
        "sample_idx": int(sample_idx),
        "true_class": str(row["health_state"]),
        "predicted_class": pred_class,
        "confidence": confidence,
        "top_features": [{"name": name, "shap_value": val} for name, val in top_features],
        "frequency_annotation": annotation.message,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed/cwru"))
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    explanation = build_sample_explanation(args.model_dir, args.processed_dir, args.sample_idx, args.top_n)
    logger.info("Sample %d: %s", args.sample_idx, explanation["frequency_annotation"])

    output_dir = ensure_dir(args.output_dir)
    save_json(explanation, output_dir / "explanation.json")
    logger.info("Saved explanation -> %s", output_dir / "explanation.json")


if __name__ == "__main__":
    main()
