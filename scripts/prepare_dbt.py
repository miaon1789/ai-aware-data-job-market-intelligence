#!/usr/bin/env python3
"""Copy narrow, text-free pipeline outputs into an isolated dbt input database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from job_market_intelligence.analytics import build_database  # noqa: E402

JOB_COLUMNS = ["job_id", "city", "analysis_role", "seniority", "posted_at"]
MENTION_COLUMNS = ["job_id", "skill", "category", "context"]


def prepare(input_dir: Path, database: Path) -> dict[str, int]:
    """Preserve the source database and avoid copying descriptions or sentences."""
    database = database.resolve()
    # Restrict writes to this dedicated directory, not the existing analysis DB.
    dedicated = (ROOT / "data/private/dbt").resolve()
    if not database.is_relative_to(dedicated):
        raise ValueError("dbt database must live under data/private/dbt")
    if database == (input_dir / "job_market.duckdb").resolve():
        raise ValueError("cannot overwrite the pipeline database")
    jobs = pd.read_csv(input_dir / "jobs_clean.csv", dtype=str, keep_default_na=False)
    mentions = pd.read_csv(input_dir / "skill_mentions.csv", dtype=str, keep_default_na=False)
    jobs = jobs[JOB_COLUMNS]
    mentions = mentions[MENTION_COLUMNS]
    audit = pd.DataFrame(columns=["removed_job_id", "kept_job_id", "similarity"])
    # This also creates the original two analytical views as independent references.
    build_database(database, jobs, mentions, audit)
    return {"jobs": len(jobs), "skill_mentions": len(mentions)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args()
    print(prepare(args.input_dir, args.database))


if __name__ == "__main__":
    main()
