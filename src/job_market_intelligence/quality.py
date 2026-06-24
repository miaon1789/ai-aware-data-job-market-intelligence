"""Aggregate-only data quality reporting for private job records."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd


def filter_recent_ads(
    jobs: pd.DataFrame, *, max_age_days: int
) -> tuple[pd.DataFrame, int, str]:
    """Filter relative to retrieval date, or dataset maximum date as fallback."""

    if max_age_days < 1:
        raise ValueError("max_age_days must be at least 1")
    posted = pd.to_datetime(jobs["posted_at"], errors="raise")
    retrieved = pd.to_datetime(jobs.get("retrieved_at", ""), errors="coerce", utc=True)
    if isinstance(retrieved, pd.Series) and retrieved.notna().any():
        reference = retrieved.max().tz_localize(None).normalize()
    else:
        reference = posted.max().normalize()
    cutoff = reference - timedelta(days=max_age_days)
    retained = jobs.loc[posted.ge(cutoff)].copy()
    return retained, len(jobs) - len(retained), reference.date().isoformat()


def build_quality_report(
    jobs: pd.DataFrame,
    skill_mentions: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    *,
    original_rows: int,
    age_filtered_rows: int,
    max_age_days: int,
    reference_date: str,
) -> dict[str, object]:
    lengths = jobs["description"].astype(str).str.len()
    words = jobs["description"].astype(str).str.split().str.len()
    exact_500_share = float(lengths.eq(500).mean()) if len(lengths) else 0.0
    excerpt_detected = exact_500_share >= 0.8
    ads_with_skills = skill_mentions["job_id"].nunique() if len(skill_mentions) else 0

    context_counts = (
        skill_mentions["context"].value_counts().sort_index().astype(int).to_dict()
        if len(skill_mentions)
        else {}
    )
    query_counts: dict[str, int] = {}
    if "collection_query" in jobs and jobs["collection_query"].astype(str).str.len().gt(0).any():
        query_counts = {
            str(key): int(value)
            for key, value in (
                jobs.assign(query=jobs["collection_query"].str.split(r" \| "))
                .explode("query")["query"]
                .value_counts()
                .sort_index()
                .items()
            )
        }
    group_counts: dict[str, int] = {}
    if "collection_group" in jobs and jobs["collection_group"].astype(str).str.len().gt(0).any():
        group_counts = {
            str(key): int(value)
            for key, value in (
                jobs.assign(group=jobs["collection_group"].str.split(r" \| "))
                .explode("group")["group"]
                .value_counts()
                .sort_index()
                .items()
            )
        }

    warnings = []
    if excerpt_detected:
        warnings.append(
            "Description truncation detected: results support excerpt-level mentions, "
            "not complete job-description requirement analysis."
        )
    if len(jobs) and ads_with_skills / len(jobs) < 0.5:
        warnings.append("Fewer than half of retained excerpts contain a dictionary skill.")

    return {
        "original_rows": original_rows,
        "age_window_days": max_age_days,
        "age_reference_date": reference_date,
        "age_filtered_rows": age_filtered_rows,
        "retained_before_deduplication": original_rows - age_filtered_rows,
        "near_duplicates_removed": int(len(duplicate_audit)),
        "clean_rows": int(len(jobs)),
        "city_counts": {
            str(key): int(value) for key, value in jobs["city"].value_counts().items()
        },
        "query_membership_counts": query_counts,
        "query_group_membership_counts": group_counts,
        "unique_companies": int(jobs["company"].nunique()),
        "posted_date_start": str(jobs["posted_at"].min()),
        "posted_date_end": str(jobs["posted_at"].max()),
        "description_characters": {
            "minimum": int(lengths.min()),
            "median": int(lengths.median()),
            "maximum": int(lengths.max()),
            "exactly_500_share": round(exact_500_share, 4),
        },
        "description_words_median": int(words.median()),
        "description_truncation_detected": excerpt_detected,
        "text_scope": "API excerpts" if excerpt_detected else "API description field",
        "skill_mentions": int(len(skill_mentions)),
        "ads_with_any_skill": int(ads_with_skills),
        "ads_with_any_skill_share": round(ads_with_skills / max(1, len(jobs)), 4),
        "skill_context_counts": context_counts,
        "warnings": warnings,
        "contains_row_level_records": False,
    }
