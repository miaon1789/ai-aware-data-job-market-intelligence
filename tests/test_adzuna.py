from urllib.parse import parse_qs, urlparse

import pytest

from job_market_intelligence.adzuna import (
    AdzunaAPIError,
    AdzunaClient,
    collect_job_ads,
)


def api_payload():
    return {
        "results": [
            {
                "id": "123",
                "title": "<b>Graduate Data Analyst</b>",
                "description": "<p>Use SQL and Power BI to build useful reports.</p>",
                "created": "2026-06-20T01:02:03Z",
                "company": {"display_name": "Example Analytics"},
                "redirect_url": "https://www.adzuna.com.au/details/123",
            },
            {"id": "bad", "title": "X", "description": "too short"},
        ]
    }


def test_client_builds_australian_query_without_logging_credentials():
    observed = {}

    def request_json(url: str, timeout: float):
        observed["url"] = url
        observed["timeout"] = timeout
        return api_payload()

    client = AdzunaClient("test-id", "test-key", request_json=request_json)
    payload = client.search(query="data analyst", city="Sydney", page=1)
    parsed = urlparse(observed["url"])
    parameters = parse_qs(parsed.query)
    assert parsed.path.endswith("/jobs/au/search/1")
    assert parameters["what"] == ["data analyst"]
    assert parameters["where"] == ["Sydney"]
    assert parameters["app_id"] == ["test-id"]
    assert payload["results"]


def test_collection_rate_limits_normalises_and_deduplicates():
    calls = []
    sleeps = []

    def request_json(url: str, timeout: float):
        calls.append(url)
        return api_payload()

    result = collect_job_ads(
        AdzunaClient("id", "key", request_json=request_json),
        queries=["data analyst"],
        cities=["Sydney"],
        pages_per_query=2,
        request_delay_seconds=2.4,
        sleep=sleeps.append,
    )
    assert result.calls == 2
    assert result.received_records == 4
    assert result.skipped_records == 2
    assert len(result.jobs) == 1
    assert result.jobs.loc[0, "description"] == "Use SQL and Power BI to build useful reports."
    assert result.jobs.loc[0, "role_label"] == ""
    assert result.jobs.loc[0, "collection_group"] == "custom"
    assert sleeps == [2.4]
    assert len(calls) == 2


def test_collection_tracks_query_groups_for_deduplicated_ads():
    def request_json(url: str, timeout: float):
        return api_payload()

    result = collect_job_ads(
        AdzunaClient("id", "key", request_json=request_json),
        queries={"entry_level": ["graduate data analyst"], "ai_signal": ["AI analyst"]},
        cities=["Sydney"],
        request_delay_seconds=2.4,
        sleep=lambda seconds: None,
    )

    assert result.calls == 2
    assert len(result.jobs) == 1
    assert result.jobs.loc[0, "collection_query"] == "AI analyst | graduate data analyst"
    assert result.jobs.loc[0, "collection_group"] == "ai_signal | entry_level"


def test_collection_rejects_unsafe_rate_and_malformed_response():
    client = AdzunaClient("id", "key", request_json=lambda url, timeout: {"unexpected": []})
    with pytest.raises(ValueError, match="at least 2.4"):
        collect_job_ads(client, request_delay_seconds=0)
    with pytest.raises(AdzunaAPIError, match="results list"):
        client.search(query="data analyst", city="Sydney", page=1)


def test_collection_retries_transient_api_errors():
    attempts = []
    sleeps = []

    def request_json(url: str, timeout: float):
        attempts.append(url)
        if len(attempts) == 1:
            raise AdzunaAPIError("temporary")
        return api_payload()

    result = collect_job_ads(
        AdzunaClient("id", "key", request_json=request_json),
        queries=["data analyst"],
        cities=["Sydney"],
        request_delay_seconds=2.4,
        sleep=sleeps.append,
    )

    assert result.calls == 1
    assert len(attempts) == 2
    assert sleeps == [2.4]
