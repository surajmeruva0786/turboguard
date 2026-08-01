#!/usr/bin/env python
"""Evaluate RUL estimation (direct regression, health-indicator + degradation
model, and the combined ensemble) on the synthetic IMS run-to-failure
trajectories (README section 12, roadmap step 80).

Bearing 1 is used to fit the health-indicator autoencoder (its earliest,
healthiest snapshots) and to learn the ensemble blend weight; bearing 2 is
held out as the evaluation trajectory. Direct regression reuses the
smoke-trained hybrid checkpoint (``runs/hybrid_smoke/model.pt``) — note
that checkpoint was trained on this same synthetic data (see
``docs/STATUS.md``), so these are pipeline-correctness numbers, not a
held-out benchmark claim.

Usage:
    python -m src.evaluation.evaluate_rul --output_dir results/rul_ims
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.cwru_loader import align_channels
from src.data.ims_loader import IMSRecord, load_ims_dataset
from src.evaluation.rul_metrics import rul_metrics
from src.features.bearing_freqs import IMS_BEARING, IMS_SHAFT_FREQ_HZ
from src.features.feature_vector import extract_feature_vector, feature_dict_to_array
from src.rul.combine import combine_rul_estimates, learn_combination_weight
from src.rul.degradation_model import estimate_rul_from_trajectory
from src.rul.direct_regression import load_direct_regression_model, predict_rul
from src.rul.health_indicator import fit_health_indicator_model
from src.utils.io import ensure_dir, save_json, save_yaml
from src.utils.logging_config import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

SEQ_LEN = 10
N_HEALTHY_SNAPSHOTS = 5


def extract_bearing_features(records: list[IMSRecord], n_channels: int = 3) -> np.ndarray:
    records = sorted(records, key=lambda r: r.snapshot_index)
    feature_dicts = []
    for r in records:
        signal = align_channels(r.signal, (), n_channels=n_channels)
        feats = extract_feature_vector(
            signal, r.sample_rate_hz, shaft_freq_hz=IMS_SHAFT_FREQ_HZ, geom=IMS_BEARING
        )
        feature_dicts.append(feats)
    X, _ = feature_dict_to_array(feature_dicts)
    return X


def build_rul_sequences(records: list[IMSRecord], seq_len: int, n_channels: int = 3):
    """Sliding windows of ``seq_len`` consecutive snapshots -> (windows, actual_rul, end_index)."""
    records = sorted(records, key=lambda r: r.snapshot_index)
    signals = [align_channels(r.signal, (), n_channels=n_channels).astype(np.float32) for r in records]
    windows, actual, end_indices = [], [], []
    for end in range(seq_len - 1, len(signals)):
        start = end - seq_len + 1
        windows.append(np.stack(signals[start : end + 1]))
        actual.append(float(records[end].rul_cycles))
        end_indices.append(records[end].snapshot_index)
    return np.stack(windows), np.array(actual), np.array(end_indices)


def degradation_rul_at_indices(
    time_index: np.ndarray, hi_values: np.ndarray, threshold: float, eval_indices: np.ndarray
) -> np.ndarray:
    preds = []
    for i in eval_indices:
        i = int(i)
        preds.append(estimate_rul_from_trajectory(time_index[: i + 1], hi_values[: i + 1], threshold))
    return np.array(preds)


def evaluate_bearing_pair(fit_bearing: int, test_bearing: int, ims_dir: Path, hybrid_checkpoint: Path) -> dict:
    fit_records = load_ims_dataset(ims_dir, bearing_id=fit_bearing)
    test_records = load_ims_dataset(ims_dir, bearing_id=test_bearing)

    fit_records = sorted(fit_records, key=lambda r: r.snapshot_index)
    test_records = sorted(test_records, key=lambda r: r.snapshot_index)

    X_fit = extract_bearing_features(fit_records)
    X_test = extract_bearing_features(test_records)

    healthy_X = X_fit[:N_HEALTHY_SNAPSHOTS]
    hi_model = fit_health_indicator_model(healthy_X, latent_dim=4, hidden_dims=(16,), epochs=200)

    hi_fit = hi_model.compute(X_fit)
    hi_test = hi_model.compute(X_test)
    threshold = float(hi_fit.max())

    time_fit = np.array([r.snapshot_index for r in fit_records], dtype=np.float64)
    time_test = np.array([r.snapshot_index for r in test_records], dtype=np.float64)

    # seq_len history is needed for direct regression, so evaluation indices start there
    eval_indices = np.arange(SEQ_LEN - 1, len(test_records))
    hi_preds_test = degradation_rul_at_indices(time_test, hi_test, threshold, eval_indices)
    hi_preds_fit = degradation_rul_at_indices(time_fit, hi_fit, threshold, eval_indices)

    model = load_direct_regression_model(hybrid_checkpoint)
    windows_fit, actual_fit, end_idx_fit = build_rul_sequences(fit_records, SEQ_LEN)
    windows_test, actual_test, end_idx_test = build_rul_sequences(test_records, SEQ_LEN)
    assert np.array_equal(end_idx_fit, eval_indices)
    assert np.array_equal(end_idx_test, eval_indices)

    direct_preds_fit = predict_rul(model, windows_fit)
    direct_preds_test = predict_rul(model, windows_test)

    weight_direct = learn_combination_weight(direct_preds_fit, hi_preds_fit, actual_fit)
    combined_preds_test = combine_rul_estimates(direct_preds_test, hi_preds_test, weight_direct)

    return {
        "fit_bearing": fit_bearing,
        "test_bearing": test_bearing,
        "weight_direct": weight_direct,
        "threshold": threshold,
        "health_indicator_only": rul_metrics(actual_test, hi_preds_test),
        "direct_regression_only": rul_metrics(actual_test, direct_preds_test),
        "combined": rul_metrics(actual_test, combined_preds_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ims_dir", type=Path, default=Path("data/raw/ims/synthetic"))
    parser.add_argument("--hybrid_checkpoint", type=Path, default=Path("runs/hybrid_smoke/model.pt"))
    parser.add_argument("--fit_bearing", type=int, default=1)
    parser.add_argument("--test_bearing", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("results/rul_ims"))
    args = parser.parse_args()

    set_seed(args.seed)
    results = evaluate_bearing_pair(args.fit_bearing, args.test_bearing, args.ims_dir, args.hybrid_checkpoint)
    logger.info(
        "RUL eval (bearing %d -> %d): HI RMSE=%.2f, Direct RMSE=%.2f, Combined RMSE=%.2f",
        args.fit_bearing,
        args.test_bearing,
        results["health_indicator_only"]["rmse"],
        results["direct_regression_only"]["rmse"],
        results["combined"]["rmse"],
    )

    output_dir = ensure_dir(args.output_dir)
    save_json(results, output_dir / "metrics.json")
    save_yaml(
        {
            "note": (
                "Evaluated on the tiny 2-bearing synthetic IMS set; direct-regression "
                "checkpoint was smoke-trained on this same data (not held out) — this is "
                "a pipeline-correctness check, not a benchmark claim."
            ),
            "fit_bearing": args.fit_bearing,
            "test_bearing": args.test_bearing,
            "seq_len": SEQ_LEN,
            "n_healthy_snapshots": N_HEALTHY_SNAPSHOTS,
        },
        output_dir / "config.yaml",
    )
    logger.info("Saved RUL evaluation -> %s", output_dir)


if __name__ == "__main__":
    main()
