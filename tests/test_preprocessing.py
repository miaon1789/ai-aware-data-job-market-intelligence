import pandas as pd

from job_market_intelligence.preprocessing import deduplicate_job_ads, redact_personal_contacts


def test_deduplication_keeps_newest_near_duplicate():
    description = "Build SQL reports and Power BI dashboards for operational stakeholders."
    frame = pd.DataFrame(
        [
            {
                "job_id": "old",
                "title": "Data Analyst",
                "description": description,
                "company": "Example",
                "city": "Sydney",
                "posted_at": "2026-01-01",
            },
            {
                "job_id": "new",
                "title": "Data Analyst",
                "description": description + " ",
                "company": "Example",
                "city": "Sydney",
                "posted_at": "2026-01-02",
            },
        ]
    )
    cleaned, audit = deduplicate_job_ads(frame)
    assert cleaned["job_id"].tolist() == ["new"]
    assert audit.loc[0, "removed_job_id"] == "old"
    assert audit.loc[0, "kept_job_id"] == "new"


def test_redacts_email_and_australian_phone_number():
    text = "Contact recruiter@example.com or +61 412 345 678 for details."
    result = redact_personal_contacts(text)
    assert "recruiter@example.com" not in result
    assert "+61 412 345 678" not in result
    assert result == "Contact [REDACTED_EMAIL] or [REDACTED_PHONE] for details."
