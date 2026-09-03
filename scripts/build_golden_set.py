#!/usr/bin/env python3
"""Resolve the committed golden queries against the private corpus.

Two artefacts come out of this. The private one carries real advertisement ids
and stays under `data/private/`. The published one carries only hashed ids and
per-query counts, which is enough to verify the evaluation without exposing
which licensed advertisements were retrieved.
"""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from job_market_intelligence.retrieval.golden import (  # noqa: E402
    document_hash,
    judged_documents,
    load_queries,
    resolve_relevance,
)

PRIVATE_ROOT = (ROOT / "data/private").resolve()
EVAL_ROOT = (ROOT / "eval").resolve()


def inside(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    return resolved == root or root in resolved.parents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=ROOT / "eval/golden/queries.jsonl")
    parser.add_argument(
        "--jobs", type=Path, default=ROOT / "data/private/retrieval/jobs_clean.csv"
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/private/retrieval/job_market.duckdb"
    )
    parser.add_argument(
        "--document-labels",
        type=Path,
        default=ROOT
        / "data/private/annotations_expanded/processed_adjudicated/document_labels.csv",
    )
    parser.add_argument(
        "--private-output",
        type=Path,
        default=ROOT / "data/private/retrieval/golden/resolved.jsonl",
    )
    parser.add_argument(
        "--published-output", type=Path, default=ROOT / "eval/golden/relevance_hashed.jsonl"
    )
    parser.add_argument("--summary", type=Path, default=ROOT / "eval/golden/summary.json")
    args = parser.parse_args()

    if not inside(args.private_output, PRIVATE_ROOT):
        parser.error("resolved advertisement ids must stay under data/private")
    if not inside(args.published_output, EVAL_ROOT):
        parser.error("published relevance must be written under eval/")
    for path in (args.queries, args.jobs, args.database, args.document_labels):
        if not path.exists():
            parser.error(f"missing input: {path}")

    golden = load_queries(args.queries)
    jobs = pd.read_csv(args.jobs, dtype=str, keep_default_na=False)
    document_labels = pd.read_csv(args.document_labels, dtype=str, keep_default_na=False)
    with duckdb.connect(str(args.database), read_only=True) as connection:
        skill_mentions = connection.execute(
            "SELECT job_id, skill, category, context FROM skill_mentions"
        ).fetchdf()
    skill_mentions["job_id"] = skill_mentions["job_id"].astype(str)

    private_rows: list[dict[str, object]] = []
    published_rows: list[dict[str, object]] = []
    empty: list[str] = []
    for query in golden:
        relevant = resolve_relevance(
            query,
            jobs=jobs,
            skill_mentions=skill_mentions,
            document_labels=document_labels,
        )
        pool = judged_documents(query, jobs=jobs, document_labels=document_labels)
        if pool is not None:
            relevant &= pool
        if not relevant:
            empty.append(query.query_id)

        ordered = sorted(relevant)
        private_rows.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "query_class": query.query_class,
                "judged_pool": query.judged_pool,
                "judged_documents": len(jobs) if pool is None else len(pool),
                # The judged pool is every advertisement this query has a verdict
                # for, relevant or not; scoring condenses rankings onto it.
                "judged_job_ids": [] if pool is None else sorted(pool),
                "relevant_job_ids": ordered,
            }
        )
        published_rows.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "query_class": query.query_class,
                "judged_pool": query.judged_pool,
                "judged_documents": len(jobs) if pool is None else len(pool),
                "relevant_documents": len(ordered),
                "relevant_document_hashes": sorted(document_hash(job_id) for job_id in ordered),
            }
        )

    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_output.parent.chmod(0o700)
    args.private_output.write_text(
        "\n".join(json.dumps(row) for row in private_rows) + "\n", encoding="utf-8"
    )
    args.private_output.chmod(0o600)

    args.published_output.parent.mkdir(parents=True, exist_ok=True)
    args.published_output.write_text(
        "\n".join(json.dumps(row) for row in published_rows) + "\n", encoding="utf-8"
    )

    counts = pd.DataFrame(published_rows)
    summary = {
        "queries": len(golden),
        "corpus_documents": int(len(jobs)),
        "labelled_documents": int(
            document_labels["job_id"].astype(str).isin(jobs["job_id"].astype(str)).sum()
        ),
        "by_class": {
            str(name): {
                "queries": int(len(part)),
                "median_relevant_documents": float(part["relevant_documents"].median()),
                "min_relevant_documents": int(part["relevant_documents"].min()),
                "max_relevant_documents": int(part["relevant_documents"].max()),
            }
            for name, part in counts.groupby("query_class")
        },
        "queries_with_no_relevant_documents": empty,
        "published_relevance": str(args.published_output.relative_to(ROOT)),
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
