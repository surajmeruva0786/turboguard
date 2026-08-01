import pandas as pd
from app.data_access import (
    alert_inbox,
    build_fleet_table,
    generate_maintenance_report_pdf,
    ims_bearing_trajectory,
    load_cwru_fleet,
    load_ims_fleet,
)


def test_load_cwru_fleet_returns_one_asset_per_load():
    assets = load_cwru_fleet()
    assert len(assets) == 4  # 0/1/2/3 HP
    assert all(a.dataset == "cwru" for a in assets)


def test_load_ims_fleet_returns_one_asset_per_bearing():
    assets = load_ims_fleet()
    assert len(assets) == 2  # bearing 1, bearing 2
    assert all(a.dataset == "ims" for a in assets)


def test_build_fleet_table_combines_both_sources():
    fleet = build_fleet_table()
    assert len(fleet) == 6
    assert set(fleet["dataset"]) == {"cwru", "ims"}
    assert set(fleet.columns) >= {
        "asset_id", "dataset", "predicted_class", "confidence",
        "predicted_rul_cycles", "health_color", "urgency", "message",
    }


def test_ims_bearing_trajectory_has_expected_columns_and_length():
    traj = ims_bearing_trajectory(1)
    assert len(traj) == 20  # 20 synthetic snapshots per bearing
    assert set(traj.columns) == {"snapshot_index", "rms", "kurtosis", "health_indicator", "rul_cycles"}
    assert traj["rul_cycles"].iloc[-1] == 0  # last snapshot is failure


def test_alert_inbox_excludes_routine_and_sorts_by_urgency():
    fleet = pd.DataFrame(
        [
            {"asset_id": "a", "urgency": "routine", "message": "m"},
            {"asset_id": "b", "urgency": "urgent", "message": "m"},
            {"asset_id": "c", "urgency": "immediate", "message": "m"},
        ]
    )
    alerts = alert_inbox(fleet)
    assert list(alerts["asset_id"]) == ["c", "b"]


def test_generate_maintenance_report_pdf_returns_valid_pdf_bytes():
    row = {
        "asset_id": "IMS-Bearing-1",
        "dataset": "ims",
        "predicted_class": "outer_race",
        "confidence": 0.5,
        "predicted_rul_cycles": 3.0,
        "urgency": "immediate",
        "message": "Asset IMS-Bearing-1: predicted RUL 3.0 cycles — schedule immediate inspection/shutdown.",
    }
    pdf_bytes = generate_maintenance_report_pdf(row)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 100
