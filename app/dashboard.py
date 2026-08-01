"""TurboGuard triage dashboard (README section 14).

    streamlit run app/dashboard.py

Three views (sidebar navigation): fleet overview, per-asset drill-down, and
an alert inbox with SHAP-based explanations and a downloadable maintenance
report. All data-loading / business logic lives in :mod:`app.data_access`
so it stays unit-testable without a running Streamlit session (see
``tests/test_dashboard_utils.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from app.data_access import (
    DEFAULT_IMS_DIR,
    alert_inbox,
    build_fleet_table,
    generate_maintenance_report_pdf,
    ims_bearing_trajectory,
)

st.set_page_config(page_title="TurboGuard", layout="wide")


@st.cache_data
def get_fleet() -> pd.DataFrame:
    return build_fleet_table()


def render_health_badge(color: str) -> str:
    return f":{'green' if color == 'green' else ('orange' if color == 'amber' else 'red')}[●]"


def render_fleet_view() -> None:
    st.title("TurboGuard — Fleet Overview")
    st.caption(
        "Demo fleet built from the committed synthetic CWRU/IMS datasets. "
        "CWRU assets have no run-to-failure data, so their RUL is a labelled "
        "demo heuristic, not a calibrated estimate — see `app/data_access.py`."
    )
    fleet = get_fleet()

    cols = st.columns(4)
    cols[0].metric("Assets tracked", len(fleet))
    cols[1].metric("Healthy", int((fleet["health_color"] == "green").sum()))
    cols[2].metric("Needs attention", int((fleet["health_color"] == "amber").sum()))
    cols[3].metric("Critical", int((fleet["health_color"] == "red").sum()))

    display = fleet.copy()
    display["health"] = display["health_color"].map(render_health_badge)
    st.dataframe(
        display[["health", "asset_id", "dataset", "predicted_class", "confidence", "predicted_rul_cycles", "urgency"]],
        use_container_width=True,
        hide_index=True,
    )


def render_asset_drilldown() -> None:
    st.title("Asset Drill-down")
    fleet = get_fleet()
    asset_id = st.selectbox("Asset", fleet["asset_id"])
    row = fleet[fleet["asset_id"] == asset_id].iloc[0]

    st.write(f"**Predicted class**: {row['predicted_class']} (confidence {row['confidence']:.2f})")
    st.write(f"**Predicted RUL**: {row['predicted_rul_cycles']:.1f} cycles — urgency: `{row['urgency']}`")
    st.info(row["message"])

    if row["dataset"] == "ims":
        bearing_id = int(asset_id.rsplit("-", 1)[-1])
        traj = ims_bearing_trajectory(bearing_id, DEFAULT_IMS_DIR)
        st.subheader("Run-to-failure trajectory")
        st.line_chart(traj.set_index("snapshot_index")[["rms", "kurtosis", "health_indicator"]])
    else:
        st.caption("CWRU assets are single-window snapshots; no time-series trajectory is available.")

    pdf_bytes = generate_maintenance_report_pdf(row.to_dict())
    st.download_button(
        "Download maintenance report (PDF)",
        data=pdf_bytes,
        file_name=f"{asset_id}_report.pdf",
        mime="application/pdf",
    )


def render_alert_inbox() -> None:
    st.title("Alert Inbox")
    fleet = get_fleet()
    alerts = alert_inbox(fleet)

    if alerts.empty:
        st.success("No active alerts — all assets are within routine tolerances.")
        return

    for _, row in alerts.iterrows():
        with st.expander(f"{row['asset_id']} — {row['urgency'].upper()}"):
            st.write(row["message"])
            st.caption(f"Predicted class: {row['predicted_class']} (confidence {row['confidence']:.2f})")
            st.caption(
                "Full SHAP feature-attribution waterfall for CWRU assets: "
                "`python -m src.xai.explain --model_dir runs/random_forest_cwru "
                "--processed_dir data/processed/cwru --output_dir results/xai/<asset>` "
                "(see notebooks/05_xai_walkthrough.ipynb for an inline walkthrough)."
            )


PAGES = {
    "Fleet View": render_fleet_view,
    "Asset Drill-down": render_asset_drilldown,
    "Alert Inbox": render_alert_inbox,
}


def main() -> None:
    page = st.sidebar.radio("Navigate", list(PAGES))
    PAGES[page]()


if __name__ == "__main__":
    main()
