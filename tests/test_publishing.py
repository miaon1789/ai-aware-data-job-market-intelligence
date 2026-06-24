import pandas as pd
import pytest

from job_market_intelligence.publishing import (
    PROHIBITED_PUBLIC_COLUMNS,
    build_public_release,
    release_from_database,
    write_public_release,
)


def frames(count: int, source: str = "adzuna-api"):
    jobs = pd.DataFrame(
        [
            {
                "job_id": f"job-{index}",
                "city": "Sydney",
                "analysis_role": "Data Analyst",
                "seniority": "Graduate / Junior",
                "posted_at": "2026-06-01",
                "source": source,
                "retrieved_at": "2026-06-22T01:00:00+00:00",
                "role_label": "",
                "title": f"Private title {index}",
                "description": f"Private description {index}",
                "company": f"Private company {index}",
                "source_url": f"https://example.test/{index}",
            }
            for index in range(count)
        ]
    )
    mentions = pd.DataFrame(
        [
            {
                "job_id": f"job-{index}",
                "skill": "SQL",
                "category": "Data",
                "context": "required",
                "sentence": "Private source sentence",
            }
            for index in range(count)
        ]
    )
    return jobs, mentions


def test_real_release_suppresses_small_cells_and_cannot_lower_threshold():
    jobs, mentions = frames(9)
    release = build_public_release(jobs, mentions)
    assert all(frame.empty for frame in release.tables.values())
    with pytest.raises(ValueError, match=">= 10"):
        build_public_release(jobs, mentions, minimum_cell_size=9)


def test_real_release_contains_only_aggregates_and_attribution(tmp_path):
    jobs, mentions = frames(12)
    release = build_public_release(jobs, mentions)
    assert release.tables["market_summary"].loc[0, "advertisement_count"] == 12
    assert release.tables["skill_summary"].loc[0, "ads_with_skill"] == 12
    assert release.metadata["minimum_cell_size"] == 10
    assert release.metadata["api_access_date_start"] == "2026-06-22"
    assert release.metadata["role_label_source"].startswith("unvalidated")
    assert release.metadata["contains_row_level_records"] is False
    for frame in release.tables.values():
        assert not PROHIBITED_PUBLIC_COLUMNS.intersection(frame.columns)

    write_public_release(release, tmp_path)
    combined = "\n".join(path.read_text() for path in tmp_path.iterdir())
    assert "The Adzuna API" in combined
    assert "Private description" not in combined
    assert "Private company" not in combined


def test_synthetic_release_is_explicitly_marked_and_not_suppressed():
    jobs, mentions = frames(1, source="synthetic-demo")
    release = build_public_release(jobs, mentions)
    assert release.metadata["data_kind"] == "synthetic"
    assert release.metadata["minimum_cell_size"] == 1
    assert len(release.tables["market_summary"]) == 1


def test_nullable_integer_role_labels_are_treated_as_unlabelled():
    jobs, mentions = frames(12)
    jobs["role_label"] = pd.Series([pd.NA] * len(jobs), dtype="Int32")
    release = build_public_release(jobs, mentions)
    assert release.metadata["human_role_label_share"] == 0.0
    assert release.metadata["role_label_source"].startswith("unvalidated")


def test_release_from_database_can_use_adjudicated_document_labels(tmp_path):
    jobs, mentions = frames(12)
    database = tmp_path / "jobs.duckdb"
    import duckdb

    with duckdb.connect(str(database)) as connection:
        connection.register("jobs_frame", jobs)
        connection.register("mentions_frame", mentions)
        connection.execute("CREATE TABLE jobs AS SELECT * FROM jobs_frame")
        connection.execute("CREATE TABLE skill_mentions AS SELECT * FROM mentions_frame")
    labels = pd.DataFrame(
        [
            {"job_id": f"job-{index}", "role_label": "Data Analyst / BI"}
            for index in range(10)
        ]
    )
    label_path = tmp_path / "document_labels.csv"
    labels.to_csv(label_path, index=False)

    release = release_from_database(database, document_labels_path=label_path)

    assert release.metadata["input_advertisement_count"] == 10
    assert release.metadata["role_label_source"] == "human-adjudicated semantic labels"
    assert release.tables["market_summary"].loc[0, "role"] == "Data Analyst / BI"
