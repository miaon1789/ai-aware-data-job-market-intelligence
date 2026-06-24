#!/usr/bin/env python3
"""Build a private adjudication sheet for human-vs-LLM label disagreements."""

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

from job_market_intelligence.adjudication import build_adjudication_template  # noqa: E402

PRIVATE_ROOT = (ROOT / "data/private").resolve()


def _inside_private(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == PRIVATE_ROOT or PRIVATE_ROOT in resolved.parents


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human-export",
        type=Path,
        default=ROOT
        / "data/private/annotations_expanded/exports/"
        / "documents-annotator-1.business_context_v3_candidate.jsonl",
    )
    parser.add_argument(
        "--disagreements-csv",
        type=Path,
        default=ROOT / "data/private/annotations_expanded/llm_disagreement_review.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/private/annotations_expanded/manifest.csv",
    )
    parser.add_argument(
        "--source-jobs",
        type=Path,
        default=ROOT / "data/private/adzuna/job_ads_expanded.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/private/annotations_expanded/label_adjudication_with_source.csv",
    )
    args = parser.parse_args()

    for path in [
        args.human_export,
        args.disagreements_csv,
        args.manifest,
        args.source_jobs,
        args.output,
    ]:
        if not _inside_private(path):
            parser.error("adjudication inputs and outputs must stay under data/private")
    if not args.human_export.is_file():
        parser.error(f"human export does not exist: {args.human_export}")
    if not args.disagreements_csv.is_file():
        parser.error(f"disagreement CSV does not exist: {args.disagreements_csv}")
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    if not args.source_jobs.is_file():
        parser.error(f"source jobs CSV does not exist: {args.source_jobs}")

    human_rows = _read_jsonl(args.human_export)
    with args.disagreements_csv.open(encoding="utf-8", newline="") as stream:
        disagreement_rows = list(csv.DictReader(stream))
    with args.manifest.open(encoding="utf-8", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    job_id_by_hash = {
        row["task_hash"]: row["job_id"] for row in manifest_rows if row.get("task_hash")
    }
    for row in disagreement_rows:
        row.setdefault("job_id", job_id_by_hash.get(row["task_hash"], ""))
    with args.source_jobs.open(encoding="utf-8", newline="") as stream:
        source_job_rows = list(csv.DictReader(stream))
    rows = build_adjudication_template(
        human_export_rows=human_rows,
        disagreement_rows=disagreement_rows,
        source_job_rows=source_job_rows,
    )
    if not rows:
        parser.error("no disagreement rows found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.chmod(0o600)
    print(
        json.dumps(
            {
                "adjudication_rows": len(rows),
                "output": str(args.output),
                "policy": "private review sheet; do not publish because it contains job text",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
