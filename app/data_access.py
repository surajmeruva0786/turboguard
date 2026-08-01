"""Data-access + business-logic helpers for the Streamlit dashboard (README 14).

Kept separate from :mod:`app.dashboard` so this logic is unit-testable
without importing Streamlit or running a UI event loop. Builds a small demo
"fleet" from the committed synthetic datasets: CWRU load conditions (no
run-to-failure data, so RUL there is an explicitly-labelled demo heuristic)
and the two real IMS run-to-failure trajectories (RUL = last known
ground-truth value in the trajectory, standing in for "most recent
inference" in this demo).
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.data.cwru_loader import align_channels
from src.data.dataset import FAULT_CLASSES
from src.data.ims_loader import IMSRecord, load_ims_dataset
from src.features.extract import CWRU_METADATA_COLUMNS
from src.features.time_domain import time_domain_features
from src.models.classical import ClassicalFaultClassifier
from src.utils.reports import MaintenanceRecommendation, recommend_from_rul

DEFAULT_MODEL_DIR = Path("runs/random_forest_cwru")
DEFAULT_CWRU_PROCESSED_DIR = Path("data/processed/cwru")
DEFAULT_IMS_DIR = Path("data/raw/ims/synthetic")

HEALTH_COLORS = {"green": "#2ecc71", "amber": "#f1c40f", "red": "#e74c3c"}


@dataclass
class AssetSnapshot:
    asset_id: str
    dataset: str  # "cwru" | "ims"
    predicted_class: str
    confidence: float
    predicted_rul_cycles: float
    color: str  # "green" | "amber" | "red"
    recommendation: MaintenanceRecommendation


def _color_for(predicted_class: str, rul: float, urgent_threshold: float = 5.0, lead_time: float = 20.0) -> str:
    if rul <= urgent_threshold:
        return "red"
    if predicted_class == "healthy" and rul > lead_time:
        return "green"
    return "amber"


def load_cwru_fleet(
    model_dir: Path = DEFAULT_MODEL_DIR, processed_dir: Path = DEFAULT_CWRU_PROCESSED_DIR
) -> list[AssetSnapshot]:
    """One demo "asset" per CWRU load condition, using its latest processed sample."""
    clf = ClassicalFaultClassifier.load(Path(model_dir) / "model.joblib")
    df = pd.read_parquet(Path(processed_dir) / "features.parquet")
    feature_cols = [c for c in df.columns if c not in CWRU_METADATA_COLUMNS]
    X = df[feature_cols].to_numpy(dtype=np.float64)

    assets = []
    for load_hp, group in df.groupby("load_hp"):
        idx = group.index[-1]
        pred = int(clf.predict(X[[idx]])[0])
        proba = clf.predict_proba(X[[idx]])[0]
        confidence = float(proba[list(clf.classes_).index(pred)])
        predicted_class = FAULT_CLASSES[pred]
        # No run-to-failure data for CWRU: demo-only RUL heuristic, not a calibrated estimate.
        demo_rul = 200.0 if predicted_class == "healthy" else max(5.0, 60.0 * (1 - confidence))
        asset_id = f"CWRU-Pump-{load_hp}HP"
        rec = recommend_from_rul(asset_id, demo_rul)
        assets.append(
            AssetSnapshot(
                asset_id=asset_id,
                dataset="cwru",
                predicted_class=predicted_class,
                confidence=confidence,
                predicted_rul_cycles=demo_rul,
                color=_color_for(predicted_class, demo_rul),
                recommendation=rec,
            )
        )
    return assets


def load_ims_fleet(ims_dir: Path = DEFAULT_IMS_DIR) -> list[AssetSnapshot]:
    """One asset per IMS bearing trajectory."""
    records = load_ims_dataset(ims_dir, source="synthetic")
    by_bearing: dict[int, list[IMSRecord]] = {}
    for r in records:
        by_bearing.setdefault(r.bearing_id, []).append(r)

    assets = []
    for bearing_id, recs in by_bearing.items():
        latest = sorted(recs, key=lambda r: r.snapshot_index)[-1]
        asset_id = f"IMS-Bearing-{bearing_id}"
        rec = recommend_from_rul(asset_id, float(latest.rul_cycles))
        assets.append(
            AssetSnapshot(
                asset_id=asset_id,
                dataset="ims",
                predicted_class=latest.dominant_fault,
                confidence=float(np.clip(1.0 - latest.health_indicator, 0.0, 1.0)),
                predicted_rul_cycles=float(latest.rul_cycles),
                color=_color_for(latest.dominant_fault, float(latest.rul_cycles)),
                recommendation=rec,
            )
        )
    return assets


def build_fleet_table(
    model_dir: Path = DEFAULT_MODEL_DIR,
    cwru_processed_dir: Path = DEFAULT_CWRU_PROCESSED_DIR,
    ims_dir: Path = DEFAULT_IMS_DIR,
) -> pd.DataFrame:
    """The full demo fleet as a flat DataFrame, ready for the fleet-view table."""
    assets = load_cwru_fleet(model_dir, cwru_processed_dir) + load_ims_fleet(ims_dir)
    return pd.DataFrame(
        [
            {
                "asset_id": a.asset_id,
                "dataset": a.dataset,
                "predicted_class": a.predicted_class,
                "confidence": round(a.confidence, 3),
                "predicted_rul_cycles": round(a.predicted_rul_cycles, 1),
                "health_color": a.color,
                "urgency": a.recommendation.urgency,
                "message": a.recommendation.message,
            }
            for a in assets
        ]
    )


def ims_bearing_trajectory(bearing_id: int, ims_dir: Path = DEFAULT_IMS_DIR) -> pd.DataFrame:
    """Per-snapshot RMS/kurtosis/health-indicator/RUL for one IMS bearing —
    the asset drill-down page's time-series view (README 14)."""
    records = sorted(load_ims_dataset(ims_dir, bearing_id=bearing_id), key=lambda r: r.snapshot_index)
    rows = []
    for r in records:
        x = align_channels(r.signal, (), n_channels=3)[0]
        feats = time_domain_features(x)
        rows.append(
            {
                "snapshot_index": r.snapshot_index,
                "rms": feats["rms"],
                "kurtosis": feats["kurtosis"],
                "health_indicator": r.health_indicator,
                "rul_cycles": r.rul_cycles,
            }
        )
    return pd.DataFrame(rows)


def alert_inbox(fleet: pd.DataFrame) -> pd.DataFrame:
    """Assets needing attention (non-routine urgency), most urgent first."""
    urgency_order = {"immediate": 0, "urgent": 1, "plan_soon": 2, "routine": 3}
    alerts = fleet[fleet["urgency"] != "routine"].copy()
    alerts["_rank"] = alerts["urgency"].map(urgency_order)
    return alerts.sort_values("_rank").drop(columns="_rank")


def generate_maintenance_report_pdf(asset_row: dict) -> bytes:
    """A one-page PDF maintenance report for a single asset row (README 14,
    "Report export"). ``asset_row`` is one record from :func:`build_fleet_table`."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 72
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, y, "TurboGuard Maintenance Report")
    y -= 30

    c.setFont("Helvetica", 11)
    fields = [
        ("Asset ID", asset_row["asset_id"]),
        ("Dataset", asset_row["dataset"]),
        ("Predicted class", asset_row["predicted_class"]),
        ("Confidence", f"{asset_row['confidence']:.2f}"),
        ("Predicted RUL (cycles)", f"{asset_row['predicted_rul_cycles']:.1f}"),
        ("Urgency", asset_row["urgency"]),
    ]
    for label, value in fields:
        c.drawString(72, y, f"{label}: {value}")
        y -= 20

    y -= 10
    c.setFont("Helvetica-Oblique", 10)
    for line in _wrap_text(asset_row["message"], 90):
        c.drawString(72, y, line)
        y -= 15

    c.showPage()
    c.save()
    return buffer.getvalue()


def _wrap_text(text: str, width_chars: int) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines
