"""Build non-row-level public artefacts from a private analysis database."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from .adzuna import ATTRIBUTION_URL, TERMS_URL

MINIMUM_REAL_CELL_SIZE = 10
PUBLIC_TABLE_COLUMNS = {
    "market_summary": ["city", "role", "collection_month", "advertisement_count"],
    "seniority_summary": ["city", "role", "seniority", "advertisement_count"],
    "skill_summary": [
        "city",
        "role",
        "skill",
        "category",
        "context",
        "ads_with_skill",
        "role_city_advertisements",
        "share_of_advertisements",
    ],
}
PROHIBITED_PUBLIC_COLUMNS = {
    "job_id",
    "title",
    "description",
    "company",
    "source_url",
    "sentence",
    "collection_query",
}


@dataclass
class PublicRelease:
    tables: dict[str, pd.DataFrame]
    metadata: dict[str, object]


def _is_synthetic(jobs: pd.DataFrame) -> bool:
    sources = jobs["source"].dropna().astype(str)
    return bool(len(sources)) and sources.str.startswith("synthetic").all()


def build_public_release(
    jobs: pd.DataFrame,
    skill_mentions: pd.DataFrame,
    *,
    minimum_cell_size: int = MINIMUM_REAL_CELL_SIZE,
    role_label_source_override: str | None = None,
) -> PublicRelease:
    """Aggregate private rows and suppress small real-data cells.

    Synthetic fixtures are exempt from suppression because they contain no
    third-party or personal information. Real data can never use a threshold
    lower than ten.
    """

    synthetic = _is_synthetic(jobs)
    if not synthetic and minimum_cell_size < MINIMUM_REAL_CELL_SIZE:
        raise ValueError(
            f"real-data releases require minimum_cell_size >= {MINIMUM_REAL_CELL_SIZE}"
        )
    effective_cell_size = 1 if synthetic else minimum_cell_size

    public_jobs = jobs[
        ["job_id", "city", "analysis_role", "seniority", "posted_at", "source"]
    ].copy()
    public_jobs["retrieved_at"] = jobs.get("retrieved_at", "")
    public_jobs["collection_month"] = pd.to_datetime(
        public_jobs["posted_at"], errors="raise"
    ).dt.strftime("%Y-%m")

    market = (
        public_jobs.groupby(
            ["city", "analysis_role", "collection_month"], as_index=False
        )["job_id"]
        .nunique()
        .rename(
            columns={"analysis_role": "role", "job_id": "advertisement_count"}
        )
    )
    market = market[market["advertisement_count"] >= effective_cell_size]

    seniority = (
        public_jobs.groupby(["city", "analysis_role", "seniority"], as_index=False)[
            "job_id"
        ]
        .nunique()
        .rename(
            columns={"analysis_role": "role", "job_id": "advertisement_count"}
        )
    )
    seniority = seniority[seniority["advertisement_count"] >= effective_cell_size]

    denominators = (
        public_jobs.groupby(["city", "analysis_role"], as_index=False)["job_id"]
        .nunique()
        .rename(columns={"job_id": "role_city_advertisements"})
    )
    skill_private = skill_mentions[
        ["job_id", "skill", "category", "context"]
    ].merge(public_jobs[["job_id", "city", "analysis_role"]], on="job_id", how="inner")
    skills = (
        skill_private.groupby(
            ["city", "analysis_role", "skill", "category", "context"],
            as_index=False,
        )["job_id"]
        .nunique()
        .rename(columns={"analysis_role": "role", "job_id": "ads_with_skill"})
    )
    denominators = denominators.rename(columns={"analysis_role": "role"})
    skills = skills.merge(denominators, on=["city", "role"], how="left")
    skills = skills[
        (skills["ads_with_skill"] >= effective_cell_size)
        & (skills["role_city_advertisements"] >= effective_cell_size)
    ]
    skills["share_of_advertisements"] = (
        skills["ads_with_skill"] / skills["role_city_advertisements"]
    ).round(3)

    tables = {
        "market_summary": market[PUBLIC_TABLE_COLUMNS["market_summary"]]
        .sort_values(PUBLIC_TABLE_COLUMNS["market_summary"][:-1])
        .reset_index(drop=True),
        "seniority_summary": seniority[PUBLIC_TABLE_COLUMNS["seniority_summary"]]
        .sort_values(PUBLIC_TABLE_COLUMNS["seniority_summary"][:-1])
        .reset_index(drop=True),
        "skill_summary": skills[PUBLIC_TABLE_COLUMNS["skill_summary"]]
        .sort_values(
            ["city", "role", "ads_with_skill", "skill"],
            ascending=[True, True, False, True],
        )
        .reset_index(drop=True),
    }
    for name, frame in tables.items():
        leaked = PROHIBITED_PUBLIC_COLUMNS.intersection(frame.columns)
        if leaked:
            raise AssertionError(f"{name} contains prohibited columns: {sorted(leaked)}")

    months = public_jobs["collection_month"]
    retrieved_dates = pd.to_datetime(
        public_jobs["retrieved_at"], errors="coerce", utc=True
    ).dropna()
    accessed_start = retrieved_dates.min().date().isoformat() if len(retrieved_dates) else ""
    accessed_end = retrieved_dates.max().date().isoformat() if len(retrieved_dates) else ""
    access_text = ""
    if accessed_start:
        access_text = (
            f", accessed {accessed_start}"
            if accessed_start == accessed_end
            else f", accessed {accessed_start} to {accessed_end}"
        )
    description_lengths = jobs.get("description", pd.Series(dtype=str)).astype(str).str.len()
    exact_500_share = (
        float(description_lengths.eq(500).mean()) if len(description_lengths) else 0.0
    )
    excerpt_detected = exact_500_share >= 0.8
    role_labels = (
        jobs.get("role_label", pd.Series("", index=jobs.index))
        .astype("string")
        .fillna("")
    )
    human_label_share = float(role_labels.str.len().gt(0).mean()) if len(jobs) else 0.0
    if role_label_source_override:
        role_label_source = role_label_source_override
    elif human_label_share == 1.0:
        role_label_source = "human labels"
    elif human_label_share > 0.0:
        role_label_source = "mixed human labels and model predictions"
    else:
        role_label_source = "unvalidated synthetic-trained baseline predictions"

    metadata: dict[str, object] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "data_kind": "synthetic" if synthetic else "real aggregated API data",
        "source": "Synthetic demonstration data" if synthetic else "The Adzuna API",
        "source_url": "" if synthetic else ATTRIBUTION_URL,
        "terms_url": "" if synthetic else TERMS_URL,
        "attribution": (
            "Synthetic data generated by this project."
            if synthetic
            else (
                "Vacancy data sourced from The Adzuna API "
                f"(https://www.adzuna.com.au/){access_text}."
            )
        ),
        "minimum_cell_size": effective_cell_size,
        "real_data_minimum_cell_size_policy": MINIMUM_REAL_CELL_SIZE,
        "input_advertisement_count": int(public_jobs["job_id"].nunique()),
        "collection_month_start": str(months.min()),
        "collection_month_end": str(months.max()),
        "api_access_date_start": accessed_start,
        "api_access_date_end": accessed_end,
        "contains_row_level_records": False,
        "contains_job_descriptions": False,
        "description_truncation_detected": excerpt_detected,
        "description_exactly_500_share": round(exact_500_share, 4),
        "text_scope": "API excerpts" if excerpt_detected else "API description field",
        "role_label_source": role_label_source,
        "human_role_label_share": round(human_label_share, 4),
        "disclaimer": (
            "Aggregated results describe the collected sample, not the entire Australian "
            "labour market. No raw job advertisements are redistributed."
        ),
    }
    return PublicRelease(tables=tables, metadata=metadata)


def release_from_database(
    database_path: str | Path,
    *,
    minimum_cell_size: int = MINIMUM_REAL_CELL_SIZE,
    document_labels_path: str | Path | None = None,
) -> PublicRelease:
    with duckdb.connect(str(database_path), read_only=True) as connection:
        jobs = connection.execute(
            """
            SELECT
                job_id, city, analysis_role, seniority, posted_at, source, retrieved_at,
                description, role_label
            FROM jobs
            """
        ).fetchdf()
        mentions = connection.execute(
            "SELECT job_id, skill, category, context FROM skill_mentions"
        ).fetchdf()
    role_source_override = None
    if document_labels_path is not None:
        labels = pd.read_csv(document_labels_path, dtype=str, keep_default_na=False)
        required = {"job_id", "role_label"}
        missing = sorted(required - set(labels.columns))
        if missing:
            raise ValueError(f"document labels are missing columns: {', '.join(missing)}")
        labels = labels[["job_id", "role_label"]].drop_duplicates("job_id")
        jobs["job_id"] = jobs["job_id"].astype(str)
        labels["job_id"] = labels["job_id"].astype(str)
        jobs = jobs.merge(labels, on="job_id", how="inner", suffixes=("", "_semantic"))
        jobs["analysis_role"] = jobs["role_label_semantic"]
        jobs["role_label"] = jobs["role_label_semantic"]
        jobs = jobs.drop(columns=["role_label_semantic"])
        role_source_override = "human-adjudicated semantic labels"
    return build_public_release(
        jobs,
        mentions,
        minimum_cell_size=minimum_cell_size,
        role_label_source_override=role_source_override,
    )


def write_public_release(release: PublicRelease, output_directory: str | Path) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in release.tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "metadata.json").write_text(
        json.dumps(release.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
