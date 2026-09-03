from urllib.parse import parse_qs, urlparse

import pytest

from job_market_intelligence.jsearch import (
    JSearchAPIError,
    JSearchClient,
    collect_job_ads,
)


def api_payload():
    return {
        "status": "OK",
        "data": [
            {
                "job_id": "abc123",
                "job_title": "Graduate Data Analyst",
                "job_description": (
                    "Use SQL and Python to build dashboards and support "
                    "analytics projects across the business."
                ),
                "employer_name": "Example Analytics",
                "job_city": "Sydney",
                "job_posted_at_datetime_utc": "2026-06-20T00:00:00.000Z",
                "job_apply_link": "https://example.com/jobs/abc123",
                "job_is_remote": False,
                "job_employment_type": "FULLTIME",
                "job_min_salary": 70000,
                "job_max_salary": 85000,
                "job_salary_period": "YEAR",
            },
            {"job_id": "bad", "job_title": "X", "job_description": "too short"},
        ],
    }


def test_client_builds_located_query_with_auth_headers():
    observed = {}

    def request_json(url: str, headers: dict, timeout: float):
        observed["url"] = url
        observed["headers"] = headers
        return api_payload()

    client = JSearchClient("secret-key", request_json=request_json)
    payload = client.search(query="data analyst", location="Sydney", page=1)
    parsed = urlparse(observed["url"])
    parameters = parse_qs(parsed.query)
    assert parsed.netloc == "jsearch.p.rapidapi.com"
    assert parameters["query"] == ["data analyst in Sydney"]
    assert parameters["country"] == ["au"]
    assert observed["headers"]["X-RapidAPI-Key"] == "secret-key"
    assert observed["headers"]["X-RapidAPI-Host"] == "jsearch.p.rapidapi.com"
    assert payload["data"]


def test_collection_rate_limits_normalises_and_deduplicates():
    calls = []
    sleeps = []

    def request_json(url: str, headers: dict, timeout: float):
        calls.append(url)
        return api_payload()

    result = collect_job_ads(
        JSearchClient("key", request_json=request_json),
        queries=["data analyst"],
        cities=["Sydney"],
        pages_per_query=2,
        request_delay_seconds=1.0,
        sleep=sleeps.append,
    )
    assert result.calls == 2
    assert result.received_records == 4
    assert result.skipped_records == 2
    assert len(result.jobs) == 1
    assert "SQL and Python" in result.jobs.loc[0, "description"]
    assert result.jobs.loc[0, "role_label"] == ""
    assert result.jobs.loc[0, "collection_group"] == "custom"
    assert result.jobs.loc[0, "city"] == "Sydney"
    assert result.jobs.loc[0, "posted_at"] == "2026-06-20"
    assert result.jobs.loc[0, "salary_min"] == "70000"
    assert result.jobs.loc[0, "is_remote"] == "no"
    assert sleeps == [1.0]
    assert len(calls) == 2


def test_collection_tracks_query_groups_for_deduplicated_ads():
    def request_json(url: str, headers: dict, timeout: float):
        return api_payload()

    result = collect_job_ads(
        JSearchClient("key", request_json=request_json),
        queries={"entry_data": ["graduate data analyst"], "entry_ml": ["AI engineer"]},
        cities=["Sydney"],
        request_delay_seconds=1.0,
        sleep=lambda seconds: None,
    )

    assert result.calls == 2
    assert len(result.jobs) == 1
    assert result.jobs.loc[0, "collection_query"] == "AI engineer | graduate data analyst"
    assert result.jobs.loc[0, "collection_group"] == "entry_data | entry_ml"


def test_collection_rejects_unsafe_rate_and_malformed_response():
    client = JSearchClient("key", request_json=lambda url, headers, timeout: {"unexpected": []})
    with pytest.raises(ValueError, match="at least 1.0"):
        collect_job_ads(client, request_delay_seconds=0)
    with pytest.raises(JSearchAPIError, match="data list"):
        client.search(query="data analyst", location="Sydney", page=1)


def test_collection_retries_transient_api_errors():
    attempts = []
    sleeps = []

    def request_json(url: str, headers: dict, timeout: float):
        attempts.append(url)
        if len(attempts) == 1:
            raise JSearchAPIError("temporary")
        return api_payload()

    result = collect_job_ads(
        JSearchClient("key", request_json=request_json),
        queries=["data analyst"],
        cities=["Sydney"],
        request_delay_seconds=1.0,
        sleep=sleeps.append,
    )

    assert result.calls == 1
    assert len(attempts) == 2
    assert sleeps == [1.0]
