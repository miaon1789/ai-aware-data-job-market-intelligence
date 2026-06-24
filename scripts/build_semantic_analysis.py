#!/usr/bin/env python3
"""Build aggregate semantic-analysis tables from private labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_market_intelligence.semantic_analysis import (  # noqa: E402
    build_semantic_tables,
    suppress_small_cells,
)

PRIVATE_ROOT = (ROOT / "data/private").resolve()
REPORTS_ROOT = (ROOT / "reports").resolve()


def inside_private(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == PRIVATE_ROOT or PRIVATE_ROOT in resolved.parents


def inside_reports(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == REPORTS_ROOT or REPORTS_ROOT in resolved.parents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/processed/job_market.duckdb"
    )
    parser.add_argument(
        "--document-labels",
        type=Path,
        default=ROOT
        / "data/private/annotations_expanded/processed_business_context_v3/document_labels.csv",
    )
    parser.add_argument(
        "--private-output",
        type=Path,
        default=ROOT / "data/private/semantic_analysis",
    )
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--minimum-cell-size", type=int, default=10)
    parser.add_argument("--max-skills-per-role", type=int, default=10)
    args = parser.parse_args()

    private_output = args.private_output.resolve()
    reports_dir = args.reports_dir.resolve()
    if not inside_private(args.document_labels):
        parser.error("document labels must stay under data/private")
    if not inside_private(private_output):
        parser.error("full semantic-analysis outputs must stay under data/private")
    if not inside_reports(reports_dir):
        parser.error("aggregate semantic reports must stay under reports")
    if not args.database.exists():
        parser.error(f"analysis database does not exist: {args.database}")
    if not args.document_labels.exists():
        parser.error(f"document labels do not exist: {args.document_labels}")
    if args.minimum_cell_size < 1:
        parser.error("--minimum-cell-size must be positive")

    document_labels = pd.read_csv(args.document_labels, dtype=str, keep_default_na=False)
    with duckdb.connect(str(args.database), read_only=True) as connection:
        skill_mentions = connection.execute(
            "SELECT job_id, skill, category, context FROM skill_mentions"
        ).fetchdf()
    skill_mentions["job_id"] = skill_mentions["job_id"].astype(str)
    document_labels["job_id"] = document_labels["job_id"].astype(str)

    tables = build_semantic_tables(
        document_labels,
        skill_mentions,
        max_skills_per_role=args.max_skills_per_role,
    )

    private_output.mkdir(parents=True, exist_ok=True)
    private_output.chmod(0o700)
    reports_dir.mkdir(parents=True, exist_ok=True)

    private_paths = {}
    public_paths = {}
    for name, table in tables.items():
        private_path = private_output / f"{name}.csv"
        public_path = reports_dir / f"semantic_{name}.csv"
        table.to_csv(private_path, index=False)
        private_path.chmod(0o600)
        suppress_small_cells(
            table,
            count_column="job_count",
            minimum_cell_size=args.minimum_cell_size,
        ).to_csv(public_path, index=False)
        private_paths[name] = str(private_path.relative_to(ROOT))
        public_paths[name] = str(public_path.relative_to(ROOT))

    summary = {
        "labelled_jobs": int(document_labels["job_id"].nunique()),
        "skill_mentions": int(len(skill_mentions)),
        "document_labels": str(args.document_labels.resolve().relative_to(ROOT)),
        "private_tables": private_paths,
        "public_suppressed_tables": public_paths,
        "minimum_public_cell_size": args.minimum_cell_size,
        "methods": [
            "semantic label distribution",
            "role by AI-signal crosstab",
            "entry-fit by AI-signal crosstab",
            "role by observed-coding-signal crosstab",
            "skill frequency by semantic role",
        ],
        "privacy": (
            "Full semantic tables remain private. Public tables are aggregate-only "
            "and suppress cells below the configured threshold."
        ),
    }
    summary_path = reports_dir / "semantic_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
