#!/usr/bin/env python3
"""Summarize private LLM-audit disagreements into text-free aggregate reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_market_intelligence.llm_annotation import summarize_llm_audit_review  # noqa: E402

PRIVATE_ROOT = (ROOT / "data/private").resolve()
REPORTS_ROOT = (ROOT / "reports").resolve()


def _inside(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    return resolved == root or root in resolved.parents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=ROOT / "data/private/annotations_expanded/llm_audit_review.csv",
    )
    parser.add_argument(
        "--disagreements-csv",
        type=Path,
        default=ROOT / "data/private/annotations_expanded/llm_disagreement_review.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/human_llm_disagreement_summary.json",
    )
    args = parser.parse_args()

    if not args.review_csv.is_file():
        parser.error(f"review CSV does not exist: {args.review_csv}")
    if not _inside(args.review_csv, PRIVATE_ROOT):
        parser.error("LLM audit review input must stay under data/private")
    if not _inside(args.disagreements_csv, PRIVATE_ROOT):
        parser.error("disagreement detail output must stay under data/private")
    if not _inside(args.report, REPORTS_ROOT):
        parser.error("aggregate disagreement report must be written under reports")

    with args.review_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = summarize_llm_audit_review(rows)

    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    disagreements = [
        row for row in rows if row.get("disagreement_dimensions", "").strip()
    ]
    disagreements.sort(
        key=lambda row: (
            confidence_rank.get(row.get("confidence", ""), 99),
            row.get("disagreement_dimensions", ""),
            row.get("title", ""),
        )
    )

    args.disagreements_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.disagreements_csv.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = rows[0].keys() if rows else []
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(disagreements)
    args.disagreements_csv.chmod(0o600)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
