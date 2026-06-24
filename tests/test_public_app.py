from pathlib import Path

from streamlit.testing.v1 import AppTest

from job_market_intelligence.publishing import build_public_release, write_public_release
from tests.test_publishing import frames


def test_public_dashboard_renders_aggregate_release(tmp_path, monkeypatch):
    jobs, mentions = frames(12, source="synthetic-demo")
    write_public_release(build_public_release(jobs, mentions), tmp_path)
    monkeypatch.setenv("PUBLIC_DATA_DIR", str(tmp_path))

    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app/public_dashboard.py")).run(timeout=20)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Market overview",
        "Semantic labels",
        "Skill explorer",
        "Audit quality",
        "Governance",
    ]
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Published aggregate cells"] == "1"
