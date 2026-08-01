"""Headless dashboard smoke test (roadmap step 105).

Runs app/dashboard.py through Streamlit's ``AppTest`` harness — no browser,
no server — just verifying each page renders without raising.
"""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _run_page(page_label: str) -> AppTest:
    at = AppTest.from_file("app/dashboard.py", default_timeout=60)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value(page_label).run()
    assert not at.exception
    return at


def test_fleet_view_renders_without_error():
    at = _run_page("Fleet View")
    assert len(at.title) >= 1


def test_asset_drilldown_renders_without_error():
    at = _run_page("Asset Drill-down")
    assert len(at.title) >= 1


def test_alert_inbox_renders_without_error():
    at = _run_page("Alert Inbox")
    assert len(at.title) >= 1
