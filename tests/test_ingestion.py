from pathlib import Path

import pandas as pd
import pytest

from job_market_intelligence.ingestion import DataValidationError, load_job_ads


def valid_row():
    return {
        "job_id": "1",
        "title": "Graduate Data Analyst",
        "description": "Analyse reports using SQL and communicate useful findings.",
        "city": "sydney",
        "company": "Example Co",
        "posted_at": "2026-05-01",
        "source": "synthetic-test",
        "source_url": "",
        "role_label": "Data Analyst",
    }


def write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "jobs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_job_ads_normalises_city(tmp_path):
    frame = load_job_ads(write_csv(tmp_path, [valid_row()]))
    assert frame.loc[0, "city"] == "Sydney"
    assert frame.loc[0, "role_label"] == "Data Analyst"
    assert frame.loc[0, "collection_group"] == ""
    assert frame.loc[0, "retrieved_at"] == ""


def test_load_job_ads_reports_invalid_rows(tmp_path):
    row = valid_row()
    row["city"] = "London"
    with pytest.raises(DataValidationError, match="row 2"):
        load_job_ads(write_csv(tmp_path, [row]))
