import pandas as pd

from job_market_intelligence.quality import build_quality_report, filter_recent_ads


def test_recent_filter_uses_retrieval_date_when_available():
    jobs = pd.DataFrame(
        {
            "posted_at": ["2026-06-20", "2025-01-01"],
            "retrieved_at": ["2026-06-22T00:00:00+00:00"] * 2,
        }
    )
    retained, removed, reference = filter_recent_ads(jobs, max_age_days=90)
    assert len(retained) == 1
    assert removed == 1
    assert reference == "2026-06-22"


def test_quality_report_flags_uniform_500_character_excerpts():
    jobs = pd.DataFrame(
        [
            {
                "job_id": str(index),
                "city": "Sydney",
                "company": f"Company {index}",
                "posted_at": "2026-06-20",
                "description": "x" * 500,
                "collection_query": "data analyst",
            }
            for index in range(10)
        ]
    )
    mentions = pd.DataFrame(
        [{"job_id": "0", "context": "mentioned", "skill": "SQL"}]
    )
    duplicates = pd.DataFrame(columns=["removed_job_id", "kept_job_id", "similarity"])
    report = build_quality_report(
        jobs,
        mentions,
        duplicates,
        original_rows=10,
        age_filtered_rows=0,
        max_age_days=90,
        reference_date="2026-06-22",
    )
    assert report["description_truncation_detected"] is True
    assert report["text_scope"] == "API excerpts"
    assert report["contains_row_level_records"] is False
