import pytest

from job_market_intelligence.ats import (
    ATSAPIError,
    ATSClient,
    board_url,
    collect_ats_jobs,
)


def greenhouse_payload():
    return {
        "jobs": [
            {
                "id": 111,
                "title": "Graduate Data Scientist",
                "content": "&lt;p&gt;Build ML models with Python and SQL.&lt;/p&gt;",
                "location": {"name": "Sydney, Australia"},
                "company_name": "Databricks",
                "first_published": "2026-06-01T10:00:00+10:00",
                "absolute_url": "https://databricks.com/careers/111",
                "departments": [{"name": "Data Science"}],
            },
            {
                "id": 222,
                "title": "Data Engineer",
                "content": "Design data pipelines in Spark across large warehouses.",
                "location": {"name": "San Francisco, California"},
                "company_name": "Databricks",
                "first_published": "2026-06-02T10:00:00-07:00",
                "absolute_url": "https://databricks.com/careers/222",
            },
        ]
    }


def ashby_payload():
    return {
        "apiVersion": "1",
        "jobs": [
            {
                "id": "uuid-1",
                "title": "Junior Machine Learning Engineer",
                "descriptionPlain": "Work on ML systems and data pipelines with our Sydney team.",
                "location": "AU - Melbourne",
                "employmentType": "FullTime",
                "department": "Engineering",
                "isRemote": False,
                "publishedAt": "2026-05-20T07:48:33.401+00:00",
                "jobUrl": "https://jobs.ashbyhq.com/Airwallex/uuid-1",
            }
        ],
    }


def fake_client(routes):
    def request_json(url, timeout):
        for token, payload in routes.items():
            if token in url:
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise ATSAPIError("ATS board returned HTTP 404")

    return ATSClient(request_json=request_json)


def test_board_url_builds_expected_endpoints():
    assert board_url("greenhouse", "databricks").startswith(
        "https://boards-api.greenhouse.io/v1/boards/databricks/jobs"
    )
    assert board_url("ashby", "Airwallex") == "https://api.ashbyhq.com/posting-api/job-board/Airwallex"
    assert board_url("lever", "canva") == "https://api.lever.co/v0/postings/canva?mode=json"
    with pytest.raises(ValueError):
        board_url("bogus", "x")


def test_collect_keeps_only_au_roles_and_normalises_fields():
    client = fake_client({"databricks": greenhouse_payload(), "Airwallex": ashby_payload()})
    result = collect_ats_jobs(
        client,
        companies=[("greenhouse", "databricks", "Databricks"), ("ashby", "Airwallex", "Airwallex")],
        request_delay_seconds=0.5,
        sleep=lambda seconds: None,
    )
    assert result.received_records == 3  # 2 greenhouse + 1 ashby
    assert result.kept_records == 2  # San Francisco role dropped
    jobs = result.jobs.set_index("job_id")

    gh = jobs.loc["gh-111"]
    assert gh["city"] == "Sydney"
    assert gh["company"] == "Databricks"
    assert "Build ML models" in gh["description"]  # HTML entities decoded
    assert gh["posted_at"] == "2026-06-01"
    assert gh["source"] == "greenhouse-ats"
    assert gh["department"] == "Data Science"
    assert gh["collection_group"] == "greenhouse"

    ashby = jobs.loc["ashby-uuid-1"]
    assert ashby["city"] == "Melbourne"
    assert ashby["company"] == "Airwallex"
    assert ashby["employment_type"] == "FullTime"


def test_collect_skips_failed_boards_without_aborting():
    client = fake_client(
        {"databricks": greenhouse_payload(), "broken": ATSAPIError("HTTP 404")}
    )
    result = collect_ats_jobs(
        client,
        companies=[("greenhouse", "broken", "Broken"), ("greenhouse", "databricks", "Databricks")],
        request_delay_seconds=0.5,
        max_retries=1,
        sleep=lambda seconds: None,
    )
    assert result.companies_failed == [("greenhouse", "broken")]
    assert result.companies_fetched == 1
    assert result.kept_records == 1


def test_collect_rejects_unsafe_delay_and_bad_provider():
    client = fake_client({"databricks": greenhouse_payload()})
    with pytest.raises(ValueError, match="at least 0.5"):
        collect_ats_jobs(client, request_delay_seconds=0.1)
    with pytest.raises(ValueError, match="provider must be one of"):
        collect_ats_jobs(client, companies=[("bogus", "x")], sleep=lambda s: None)
