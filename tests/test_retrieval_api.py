"""End-to-end checks on the local FastAPI service, backed by a tiny fixture index."""

import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from job_market_intelligence.retrieval.config import RetrievalConfig  # noqa: E402
from job_market_intelligence.retrieval.lexical import LexicalIndex  # noqa: E402
from job_market_intelligence.retrieval.search import HybridRetriever  # noqa: E402

CHUNKS = pd.DataFrame(
    [
        {"chunk_id": "a::00", "job_id": "a", "chunk_index": 0, "section": "requirements",
         "heading": "Requirements", "body": "Three years with SQL and dbt.",
         "text": "Data Engineer\nRequirements\nThree years with SQL and dbt.",
         "token_estimate": 14, "title": "Data Engineer", "is_truncated": False,
         "city": "Sydney", "company": "Northwind", "posted_at": "2026-05-01",
         "source": "greenhouse-ats", "analysis_role": "Data Engineer",
         "seniority": "Non-Junior"},
        {"chunk_id": "b::00", "job_id": "b", "chunk_index": 0, "section": "intro",
         "heading": "", "body": "Build Power BI dashboards for retail stakeholders.",
         "text": "Data Analyst\nBuild Power BI dashboards for retail stakeholders.",
         "token_estimate": 15, "title": "Data Analyst", "is_truncated": True,
         "city": "Melbourne", "company": "Harbour", "posted_at": "2026-05-02",
         "source": "adzuna-api", "analysis_role": "Data Analyst", "seniority": "Unclear"},
    ]
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import api.main as main

    lexical = LexicalIndex.build(CHUNKS, tmp_path / "lex.duckdb")
    database = tmp_path / "job_market.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.register(
            "jobs_frame",
            pd.DataFrame(
                [
                    {"job_id": "a", "analysis_role": "Data Engineer", "city": "Sydney",
                     "seniority": "Non-Junior", "source": "greenhouse-ats",
                     "company": "Northwind"},
                    {"job_id": "b", "analysis_role": "Data Analyst", "city": "Melbourne",
                     "seniority": "Unclear", "source": "adzuna-api", "company": "Harbour"},
                ]
            ),
        )
        connection.register(
            "mentions_frame",
            pd.DataFrame(
                [{"job_id": "a", "skill": "SQL", "category": "Data", "context": "required"}]
            ),
        )
        connection.execute("CREATE TABLE jobs AS SELECT * FROM jobs_frame")
        connection.execute("CREATE TABLE skill_mentions AS SELECT * FROM mentions_frame")

    # Lexical-only keeps the fixture free of a model download.
    config = RetrievalConfig(mode="lexical", top_k=5, candidate_pool=5, retrieval_depth=10)
    retriever = HybridRetriever(CHUNKS, config=config, lexical_index=lexical)
    monkeypatch.setattr(main.state, "_retriever", retriever)
    monkeypatch.setattr(main.state, "config", config)
    monkeypatch.setattr(main, "stats_service", main.JobStatsService(database))
    return TestClient(main.app)


def test_health_reports_the_active_configuration(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["config"]["mode"] == "lexical"


def test_search_returns_ranked_passages_with_provenance(client):
    payload = client.post("/search", json={"query": "dbt", "top_k": 3}).json()
    assert payload["query_class"] == "lexical"
    assert payload["results"][0]["job_id"] == "a"
    assert payload["results"][0]["rank"] == 1
    assert payload["results"][0]["section"] == "requirements"


def test_search_applies_a_city_filter(client):
    payload = client.post(
        "/search", json={"query": "dashboards", "top_k": 3, "city": "Sydney"}
    ).json()
    assert all(hit["city"] == "Sydney" for hit in payload["results"])


def test_search_rejects_an_empty_query(client):
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_counting_questions_are_refused_by_the_answer_endpoint(client):
    payload = client.post("/answer", json={"question": "How many ads mention SQL?"}).json()
    assert payload["routed_to"] == "query_job_stats"
    assert "cannot count" in payload["answer"]
    assert "passages" not in payload


def test_descriptive_questions_reach_retrieval_and_stay_a_dry_run(client):
    payload = client.post(
        "/answer", json={"question": "what do these roles say about dashboards?", "top_k": 2}
    ).json()
    assert payload["routed_to"] == "search_job_ads"
    assert payload["passages"]
    assert payload["grounded_answer"]["sent"] is False


def test_stats_endpoint_counts_over_the_corpus(client):
    payload = client.post("/stats", json={"dimension": "city"}).json()
    rows = {row["value"]: row["documents"] for row in payload["rows"]}
    assert rows == {"Sydney": 1, "Melbourne": 1}


def test_stats_endpoint_rejects_an_unknown_dimension(client):
    response = client.post("/stats", json={"dimension": "salary"})
    assert response.status_code == 400
    assert "unknown dimension" in response.json()["detail"]


def test_reindex_returns_a_job_id_immediately_and_is_pollable(client, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "_reindex", lambda tokens, rebuild: f"rebuilt {tokens}")
    response = client.post("/reindex", json={"chunk_tokens": 512})
    assert response.status_code == 202
    job_id = response.json()["job"]["job_id"]

    for _ in range(200):
        status = client.get(f"/jobs/{job_id}").json()
        if status["status"] in {"succeeded", "failed"}:
            break
    assert status["status"] == "succeeded"
    assert status["finished_at"]


def test_a_failing_background_job_is_reported_not_swallowed(client, monkeypatch):
    import api.main as main

    def explode(tokens, rebuild):
        raise RuntimeError("index build failed")

    monkeypatch.setattr(main, "_reindex", explode)
    job_id = client.post("/reindex", json={}).json()["job"]["job_id"]
    for _ in range(200):
        status = client.get(f"/jobs/{job_id}").json()
        if status["status"] in {"succeeded", "failed"}:
            break
    assert status["status"] == "failed"
    assert "index build failed" in status["detail"]


def test_unknown_job_ids_are_not_found(client):
    assert client.get("/jobs/does-not-exist").status_code == 404
