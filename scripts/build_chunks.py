#!/usr/bin/env python3
"""Chunk the private advertisement corpus for retrieval.

Chunks contain licensed advertisement text and therefore stay under
`data/private/`, which the repository ignores in full.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from job_market_intelligence.retrieval.chunking import build_chunk_frame  # noqa: E402
from job_market_intelligence.retrieval.config import CHUNK_SIZES  # noqa: E402

PRIVATE_ROOT = (ROOT / "data/private").resolve()


def inside_private(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == PRIVATE_ROOT or PRIVATE_ROOT in resolved.parents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=Path,
        default=ROOT / "data/private/retrieval/jobs_clean.csv",
        help="Cleaned corpus written by run_pipeline.py",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/private/retrieval/chunks"
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        action="append",
        dest="chunk_sizes",
        help=f"Repeatable window size; defaults to {CHUNK_SIZES}",
    )
    parser.add_argument("--overlap-ratio", type=float, default=0.1)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()

    if not inside_private(args.output_dir):
        parser.error("chunk output must stay under data/private")
    if not args.jobs.exists():
        parser.error(f"jobs file does not exist: {args.jobs}")

    jobs = pd.read_csv(args.jobs, dtype=str, keep_default_na=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.chmod(0o700)

    summary: dict[str, object] = {"jobs": int(len(jobs)), "configurations": []}
    for chunk_tokens in args.chunk_sizes or list(CHUNK_SIZES):
        frame = build_chunk_frame(
            jobs, target_tokens=chunk_tokens, overlap_ratio=args.overlap_ratio
        )
        label = f"chunk{chunk_tokens}_overlap{int(args.overlap_ratio * 100)}"
        path = args.output_dir / f"{label}.parquet"
        frame.to_parquet(path, index=False)
        path.chmod(0o600)
        full_text = frame[~frame["is_truncated"]]
        summary["configurations"].append(
            {
                "label": label,
                "chunk_tokens": chunk_tokens,
                "chunks": int(len(frame)),
                "chunks_full_text": int(len(full_text)),
                "chunks_truncated": int(len(frame) - len(full_text)),
                "median_tokens_full_text": float(full_text["token_estimate"].median()),
                "chunks_per_full_text_ad": round(
                    len(full_text) / max(1, full_text["job_id"].nunique()), 2
                ),
                "path": str(path.relative_to(ROOT)),
            }
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
