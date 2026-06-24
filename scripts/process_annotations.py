#!/usr/bin/env python3
"""Validate Doccano exports and create text-free normalized private labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_market_intelligence.annotation_results import (  # noqa: E402
    AnnotationValidationError,
    process_annotation_exports,
    write_processed_annotations,
)

PRIVATE_ROOT = (ROOT / "data/private").resolve()
REPORTS_ROOT = (ROOT / "reports").resolve()


def _inside(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    return resolved == root or root in resolved.parents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/private/annotations/manifest.csv",
    )
    parser.add_argument(
        "--document-export",
        type=Path,
        action="append",
        default=[],
        help="Doccano document-classification JSONL; repeat for each annotator/export",
    )
    parser.add_argument(
        "--skill-export",
        type=Path,
        action="append",
        default=[],
        help="Doccano sequence-labeling JSONL; repeat for each annotator/export",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/private/annotations/processed",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/annotation_quality.json",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Explicitly unlock test exports after the annotation protocol is frozen",
    )
    args = parser.parse_args()

    if not args.document_export and not args.skill_export:
        parser.error("provide --document-export and/or --skill-export")
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    missing_exports = [
        path
        for path in [*args.document_export, *args.skill_export]
        if not path.is_file()
    ]
    if missing_exports:
        parser.error(f"annotation export does not exist: {missing_exports[0]}")
    if not _inside(args.output, PRIVATE_ROOT):
        parser.error("normalized annotation outputs must remain under data/private")
    if not _inside(args.report, REPORTS_ROOT):
        parser.error("the aggregate annotation report must be written under reports")

    try:
        result = process_annotation_exports(
            args.manifest,
            document_exports=args.document_export,
            skill_exports=args.skill_export,
            include_test=args.include_test,
        )
    except AnnotationValidationError as exc:
        parser.error(str(exc))
    write_processed_annotations(result, args.output, args.report)
    print(json.dumps(result.report, indent=2))


if __name__ == "__main__":
    main()
