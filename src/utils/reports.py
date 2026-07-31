"""Maintenance-report helpers used by the training/evaluation CLIs and the dashboard."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class MaintenanceRecommendation:
    """A single actionable recommendation derived from a RUL estimate."""

    asset_id: str
    predicted_rul_cycles: float
    urgency: str  # "routine" | "plan_soon" | "urgent" | "immediate"
    lead_time_cycles: float
    message: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def recommend_from_rul(
    asset_id: str,
    predicted_rul_cycles: float,
    lead_time_cycles: float = 20.0,
    urgent_threshold_cycles: float = 5.0,
) -> MaintenanceRecommendation:
    """Translate a raw RUL estimate into an urgency bucket + human-readable message.

    Policy: schedule maintenance once predicted RUL drops inside
    ``lead_time_cycles`` of failure; escalate to "urgent"/"immediate" as the
    margin shrinks further. Thresholds are intentionally simple and
    configurable — real deployments should tune them against the asset's
    criticality and spares lead time.
    """
    if predicted_rul_cycles <= urgent_threshold_cycles:
        urgency = "immediate"
        message = (
            f"Asset {asset_id}: predicted RUL {predicted_rul_cycles:.1f} cycles — "
            "schedule immediate inspection/shutdown."
        )
    elif predicted_rul_cycles <= lead_time_cycles:
        urgency = "urgent"
        message = (
            f"Asset {asset_id}: predicted RUL {predicted_rul_cycles:.1f} cycles, within "
            f"lead time ({lead_time_cycles:.0f}) — plan maintenance urgently."
        )
    elif predicted_rul_cycles <= lead_time_cycles * 2:
        urgency = "plan_soon"
        message = (
            f"Asset {asset_id}: predicted RUL {predicted_rul_cycles:.1f} cycles — "
            "add to next maintenance window."
        )
    else:
        urgency = "routine"
        message = f"Asset {asset_id}: predicted RUL {predicted_rul_cycles:.1f} cycles — no action needed."

    return MaintenanceRecommendation(
        asset_id=asset_id,
        predicted_rul_cycles=predicted_rul_cycles,
        urgency=urgency,
        lead_time_cycles=lead_time_cycles,
        message=message,
    )
