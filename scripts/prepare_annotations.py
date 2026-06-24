#!/usr/bin/env python3
"""Create private Doccano tasks with company-disjoint evaluation splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_market_intelligence.annotation import (  # noqa: E402
    prepare_annotation_package,
    write_annotation_package,
)

PRIVATE_ROOT = (ROOT / "data/private").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/processed/job_market.duckdb"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/private/annotations"
    )
    parser.add_argument("--documents", type=int, default=300)
    parser.add_argument("--skill-documents", type=int, default=150)
    parser.add_argument("--maximum-per-company", type=int, default=3)
    parser.add_argument(
        "--minimum-per-collection-group",
        type=int,
        default=0,
        help="Oversample each exact collection_group up to this minimum before filling",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = args.output.resolve()
    if PRIVATE_ROOT not in output.parents:
        parser.error("annotation packages must be stored under data/private")
    if not args.database.exists():
        parser.error("analysis database does not exist; run the pipeline first")

    with duckdb.connect(str(args.database), read_only=True) as connection:
        jobs = connection.execute(
            """
            SELECT
                job_id,
                title,
                description,
                city,
                company,
                analysis_role,
                collection_group
            FROM jobs
            """
        ).fetchdf()
    package = prepare_annotation_package(
        jobs,
        target_documents=args.documents,
        target_skill_documents=args.skill_documents,
        maximum_per_company=args.maximum_per_company,
        minimum_per_collection_group=args.minimum_per_collection_group,
        seed=args.seed,
    )
    write_annotation_package(package, output)
    print(json.dumps(package.summary, indent=2))


if __name__ == "__main__":
    main()
