#!/usr/bin/env python3
"""Create private AI-assisted first-pass document labels for Doccano review."""

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

from job_market_intelligence.annotation import task_hash  # noqa: E402
from job_market_intelligence.document_label_suggestions import (  # noqa: E402
    suggest_document_labels,
    title_from_text,
)

PRIVATE_ROOT = (ROOT / "data/private").resolve()


def inside_private(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == PRIVATE_ROOT or PRIVATE_ROOT in resolved.parents


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--document",
        type=Path,
        action="append",
        default=None,
        help="Doccano document JSONL to label; repeatable",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data/private/annotations/exports/documents-ai-suggestions.jsonl",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=ROOT / "data/private/annotations/document_label_suggestions_review.csv",
    )
    args = parser.parse_args()

    documents = args.document or [
        ROOT / "data/private/annotations/document_classification_train.jsonl",
        ROOT / "data/private/annotations/document_classification_dev.jsonl",
    ]
    for path in [*documents, args.output, args.review_csv]:
        if not inside_private(path):
            parser.error("suggestion inputs and outputs must stay under data/private")
    missing = [path for path in documents if not path.is_file()]
    if missing:
        parser.error(f"document file does not exist: {missing[0]}")

    suggestions: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for document_path in documents:
        split = document_path.stem.removeprefix("document_classification_")
        for row in read_jsonl(document_path):
            text = row.get("text")
            if not isinstance(text, str) or not text:
                parser.error(f"{document_path}: each row must contain non-empty text")
            digest = task_hash(text)
            if digest in seen:
                continue
            seen.add(digest)
            suggestion = suggest_document_labels(text)
            suggestions.append(
                {
                    "text": text,
                    "label": suggestion.labels,
                    "username": "ai-suggestion-v1",
                }
            )
            review_rows.append(
                {
                    "task_hash": digest,
                    "split": split,
                    "title": title_from_text(text),
                    "role_label": suggestion.role,
                    "entry_fit_label": suggestion.entry_fit,
                    "ai_signal_label": suggestion.ai_signal,
                    "coding_signal_label": suggestion.coding_signal,
                    "confidence": suggestion.confidence,
                    "needs_review": suggestion.needs_review,
                    "evidence": suggestion.evidence,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.chmod(0o700)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in suggestions),
        encoding="utf-8",
    )
    args.output.chmod(0o600)

    args.review_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.review_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "task_hash",
                "split",
                "title",
                "role_label",
                "entry_fit_label",
                "ai_signal_label",
                "coding_signal_label",
                "confidence",
                "needs_review",
                "evidence",
            ],
        )
        writer.writeheader()
        writer.writerows(review_rows)
    args.review_csv.chmod(0o600)

    summary = {
        "suggestions": len(suggestions),
        "output": str(args.output),
        "review_csv": str(args.review_csv),
        "policy": "AI-assisted suggestions for human review; do not report as human labels",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
