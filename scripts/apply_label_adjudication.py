#!/usr/bin/env python3
"""Apply completed private label adjudication to normalized document labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_market_intelligence.adjudication import apply_adjudications  # noqa: E402

PRIVATE_ROOT = (ROOT / "data/private").resolve()
REPORTS_ROOT = (ROOT / "reports").resolve()


def _inside(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    return resolved == root or root in resolved.parents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--document-labels",
        type=Path,
        default=ROOT
        / "data/private/annotations_expanded/processed_business_context_v3/"
        / "document_labels.csv",
    )
    parser.add_argument(
        "--adjudication-csv",
        type=Path,
        default=ROOT / "data/private/annotations_expanded/label_adjudication_with_source.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/private/annotations_expanded/processed_adjudicated",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/adjudicated_annotation_summary.json",
    )
    args = parser.parse_args()

    for path in [args.document_labels, args.adjudication_csv, args.output]:
        if not _inside(path, PRIVATE_ROOT):
            parser.error("adjudication inputs and normalized outputs must stay under data/private")
    if not _inside(args.report, REPORTS_ROOT):
        parser.error("aggregate adjudication report must be written under reports")
    if not args.document_labels.is_file():
        parser.error(f"document labels do not exist: {args.document_labels}")
    if not args.adjudication_csv.is_file():
        parser.error(f"adjudication CSV does not exist: {args.adjudication_csv}")

    document_labels = pd.read_csv(args.document_labels, dtype=str, keep_default_na=False)
    with args.adjudication_csv.open(encoding="utf-8", newline="") as stream:
        adjudication_rows = list(csv.DictReader(stream))
    try:
        final_labels, report = apply_adjudications(document_labels, adjudication_rows)
    except ValueError as exc:
        parser.error(str(exc))

    args.output.mkdir(parents=True, exist_ok=True)
    args.output.chmod(0o700)
    label_path = args.output / "document_labels.csv"
    final_labels.to_csv(label_path, index=False)
    label_path.chmod(0o600)
    skill_path = args.output / "skill_labels.jsonl"
    skill_path.write_text("", encoding="utf-8")
    skill_path.chmod(0o600)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
