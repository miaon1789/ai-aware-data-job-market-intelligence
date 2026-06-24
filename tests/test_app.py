from pathlib import Path

from streamlit.testing.v1 import AppTest

from scripts.generate_synthetic_data import generate_csv
from scripts.run_pipeline import run


def test_streamlit_app_renders_without_exceptions(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    synthetic = generate_csv(tmp_path / "synthetic/job_ads.csv")
    processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    run(synthetic, synthetic, processed, reports)
    monkeypatch.setenv("PRIVATE_DATABASE", str(processed / "job_market.duckdb"))
    monkeypatch.setenv("PRIVATE_MODEL_PATH", str(processed / "role_classifier.joblib"))
    monkeypatch.setenv("PRIVATE_METRICS_PATH", str(reports / "model_metrics.json"))
    app = AppTest.from_file(str(root / "app/streamlit_app.py")).run(timeout=20)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Market overview",
        "Skill explorer",
        "JD comparison",
        "Methods",
    ]
    app.text_input[0].input("Graduate Data Engineer")
    app.text_area[0].input(
        "Essential skills are Python, SQL, dbt and data pipelines. "
        "Experience with AWS and Airflow is desirable. This is a graduate role."
    )
    app.button[0].click().run(timeout=20)
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Predicted role"] == "Data Engineer"
    assert metrics["Seniority signal"] == "Graduate / Junior"
