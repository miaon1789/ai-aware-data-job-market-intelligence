#!/usr/bin/env python3
"""Merge multiple job-ad CSVs (Adzuna, ATS, ...) into one input for the pipeline.

Aligns columns across sources, deduplicates by job_id (first source wins), and
writes a combined private CSV to feed `run_pipeline.py --input`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "job_id", "title", "description", "city",
    "company", "posted_at", "source", "source_url",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, dest="inputs", required=True,
                        help="Source CSV; repeatable. Earlier inputs win on job_id collisions.")
    parser.add_argument("--output", type=Path, default=ROOT / "data/private/merged/job_ads.csv")
    args = parser.parse_args()

    frames = []
    for path in args.inputs:
        if not path.exists():
            parser.error(f"input not found: {path}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = [column for column in REQUIRED if column not in frame.columns]
        if missing:
            parser.error(f"{path} is missing columns: {', '.join(missing)}")
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    before = len(combined)
    combined = combined.drop_duplicates(subset="job_id", keep="first").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.chmod(0o700)
    combined.to_csv(args.output, index=False)

    print(
        f"merged {len(args.inputs)} sources: {before} rows -> {len(combined)} unique "
        f"({before - len(combined)} duplicates removed)\nwrote {args.output}",
        file=sys.stderr,
    )
    print(str(args.output))


if __name__ == "__main__":
    main()
