import pandas as pd

from scripts.merge_adzuna_snapshots import merge_snapshots


def row(job_id, query, group):
    return {
        "job_id": job_id,
        "title": "Graduate Data Analyst",
        "description": "Analyse SQL reports and dashboards.",
        "city": "Sydney",
        "company": "Example",
        "posted_at": "2026-06-01",
        "source": "adzuna-api",
        "source_url": "",
        "role_label": "",
        "collection_query": query,
        "collection_group": group,
        "retrieved_at": "2026-06-23T00:00:00+00:00",
    }


def test_merge_snapshots_deduplicates_and_combines_query_evidence(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame([row("1", "junior data analyst", "entry_level")]).to_csv(
        first, index=False
    )
    pd.DataFrame(
        [
            row("1", "data analyst Copilot", "ai_signal"),
            row("2", "prompt engineer", "ai_signal"),
        ]
    ).to_csv(second, index=False)

    merged = merge_snapshots([first, second])

    assert len(merged) == 2
    duplicate = merged.loc[merged["job_id"] == "1"].iloc[0]
    assert duplicate["collection_query"] == "data analyst Copilot | junior data analyst"
    assert duplicate["collection_group"] == "ai_signal | entry_level"


def test_merge_snapshots_backfills_legacy_collection_group(tmp_path):
    legacy = tmp_path / "legacy.csv"
    data = row("1", "data analyst", "general_baseline")
    data.pop("collection_group")
    pd.DataFrame([data]).to_csv(legacy, index=False)

    merged = merge_snapshots([legacy])

    assert merged.loc[0, "collection_group"] == "general_baseline"
