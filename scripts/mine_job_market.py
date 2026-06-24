#!/usr/bin/env python3
"""Run clustering and association-rule mining on the private analysis database."""

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

from job_market_intelligence.mining import (  # noqa: E402
    cluster_job_texts,
    mine_skill_association_rules,
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
        "--private-output",
        type=Path,
        default=ROOT / "data/private/mining",
        help="Private row-level mining outputs",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports",
        help="Text-free aggregate mining reports",
    )
    parser.add_argument(
        "--document-labels",
        type=Path,
        help="Optional private processed document_labels.csv for improved role summaries",
    )
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--cluster-terms", type=int, default=8)
    parser.add_argument("--min-support", type=float, default=0.005)
    parser.add_argument("--min-confidence", type=float, default=0.2)
    parser.add_argument("--max-rules", type=int, default=100)
    parser.add_argument(
        "--minimum-cell-size",
        type=int,
        default=10,
        help="Minimum count for aggregate outputs written under reports",
    )
    args = parser.parse_args()

    private_output = args.private_output.resolve()
    reports_dir = args.reports_dir.resolve()
    if not inside_private(private_output):
        parser.error("row-level mining outputs must stay under data/private")
    if not inside_reports(reports_dir):
        parser.error("aggregate mining reports must stay under reports")
    if args.document_labels and not inside_private(args.document_labels):
        parser.error("document label inputs must stay under data/private")
    if not args.database.exists():
        parser.error(f"analysis database does not exist: {args.database}")
    if args.document_labels and not args.document_labels.is_file():
        parser.error(f"document labels do not exist: {args.document_labels}")

    with duckdb.connect(str(args.database), read_only=True) as connection:
        jobs = connection.execute(
            """
            SELECT
                job_id,
                title,
                description,
                city,
                analysis_role,
                seniority
            FROM jobs
            """
        ).fetchdf()
        skill_mentions = connection.execute(
            "SELECT job_id, skill, category, context FROM skill_mentions"
        ).fetchdf()
    label_source = "database analysis_role"
    if args.document_labels:
        labels = pd.read_csv(args.document_labels, dtype=str, keep_default_na=False)
        required = {"job_id", "role_label"}
        missing = sorted(required - set(labels.columns))
        if missing:
            parser.error(f"document labels are missing columns: {', '.join(missing)}")
        role_labels = (
            labels[["job_id", "role_label"]]
            .drop_duplicates(["job_id", "role_label"])
            .groupby("job_id", as_index=False)["role_label"]
            .agg(lambda values: values.value_counts().index[0])
        )
        jobs = jobs.merge(role_labels, on="job_id", how="left")
        jobs["analysis_role"] = jobs["role_label"].where(
            jobs["role_label"].fillna("").ne(""), jobs["analysis_role"]
        )
        jobs = jobs.drop(columns=["role_label"])
        label_source = str(args.document_labels.resolve().relative_to(ROOT))

    cluster_result = cluster_job_texts(
        jobs,
        n_clusters=args.clusters,
        terms_per_cluster=args.cluster_terms,
    )
    rules = mine_skill_association_rules(
        skill_mentions,
        total_jobs=len(jobs),
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        max_rules=args.max_rules,
    )

    private_output.mkdir(parents=True, exist_ok=True)
    private_output.chmod(0o700)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.minimum_cell_size < 1:
        parser.error("--minimum-cell-size must be positive")

    cluster_assignments_path = private_output / "job_clusters.csv"
    private_cluster_summary_path = private_output / "job_cluster_summary.csv"
    private_rules_path = private_output / "skill_association_rules.csv"
    cluster_result.assignments.to_csv(cluster_assignments_path, index=False)
    cluster_result.summary.to_csv(private_cluster_summary_path, index=False)
    rules.to_csv(private_rules_path, index=False)
    cluster_assignments_path.chmod(0o600)
    private_cluster_summary_path.chmod(0o600)
    private_rules_path.chmod(0o600)

    cluster_summary_path = reports_dir / "job_cluster_summary.csv"
    rules_path = reports_dir / "skill_association_rules.csv"
    summary_path = reports_dir / "mining_summary.json"
    public_cluster_summary = cluster_result.summary[
        cluster_result.summary["job_count"] >= args.minimum_cell_size
    ][["cluster_id", "job_count", "share_of_jobs", "top_role", "top_seniority", "top_city"]]
    public_rules = rules[rules["job_count"] >= args.minimum_cell_size]
    public_cluster_summary.to_csv(cluster_summary_path, index=False)
    public_rules.to_csv(rules_path, index=False)

    summary = {
        "input_jobs": int(len(jobs)),
        "skill_mentions": int(len(skill_mentions)),
        "clusters": int(len(cluster_result.summary)),
        "cluster_assignments_private": str(
            cluster_assignments_path.relative_to(ROOT)
        ),
        "cluster_summary_private": str(private_cluster_summary_path.relative_to(ROOT)),
        "association_rules_private": str(private_rules_path.relative_to(ROOT)),
        "cluster_summary": str(cluster_summary_path.relative_to(ROOT)),
        "association_rules": str(rules_path.relative_to(ROOT)),
        "private_association_rule_count": int(len(rules)),
        "public_association_rule_count": int(len(public_rules)),
        "role_label_source": label_source,
        "minimum_public_cell_size": args.minimum_cell_size,
        "min_support": args.min_support,
        "min_confidence": args.min_confidence,
        "methods": [
            "TF-IDF + KMeans job-family discovery",
            "pairwise association-rule mining over extracted skills",
        ],
        "privacy": (
            "Cluster assignments remain private. Public reports contain no job IDs, "
            "titles, descriptions, companies, URLs or source sentences."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
