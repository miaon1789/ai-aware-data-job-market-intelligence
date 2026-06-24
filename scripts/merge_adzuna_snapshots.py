#!/usr/bin/env python3
"""Merge private Adzuna CSV snapshots without publishing row-level text."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_market_intelligence.ingestion import REQUIRED_COLUMNS  # noqa: E402

PRIVATE_ROOT = (ROOT / "data/private").resolve()
PROVENANCE_COLUMNS = ["collection_query", "collection_group", "retrieved_at"]


def _inside(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    return resolved == root or root in resolved.parents


def _merge_terms(values: pd.Series) -> str:
    terms: set[str] = set()
    for value in values.dropna().astype(str):
        terms.update(term.strip() for term in value.split(" | ") if term.strip())
    return " | ".join(sorted(terms))


def merge_snapshots(paths: list[Path]) -> pd.DataFrame:
    frames = []
    required = REQUIRED_COLUMNS | set(PROVENANCE_COLUMNS)
    for path in paths:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "collection_group" not in frame.columns:
            frame["collection_group"] = "general_baseline"
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        frame["_source_file"] = path.name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return combined.drop(columns=["_source_file"], errors="ignore")
    combined = combined.sort_values(["job_id", "retrieved_at", "_source_file"])
    first_rows = combined.drop_duplicates("job_id", keep="first").set_index("job_id")
    for column in ("collection_query", "collection_group"):
        first_rows[column] = combined.groupby("job_id")[column].agg(_merge_terms)
    first_rows["retrieved_at"] = combined.groupby("job_id")["retrieved_at"].max()
    return (
        first_rows.reset_index()
        .drop(columns=["_source_file"], errors="ignore")
        .sort_values("job_id")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/private/adzuna/job_ads_expanded.csv",
    )
    args = parser.parse_args()

    paths = [path.resolve() for path in args.inputs]
    output = args.output.resolve()
    for path in [*paths, output]:
        if not _inside(path, PRIVATE_ROOT):
            parser.error("inputs and output must stay under data/private")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        parser.error(f"input does not exist: {missing[0]}")

    merged = merge_snapshots(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.parent.chmod(0o700)
    merged.to_csv(output, index=False)
    output.chmod(0o600)

    provenance = {
        "merged_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "inputs": [str(path) for path in paths],
        "input_rows": int(sum(pd.read_csv(path, dtype=str).shape[0] for path in paths)),
        "unique_records": int(len(merged)),
        "dedupe_key": "job_id",
        "publication_policy": "private row-level merge; public releases remain aggregate only",
    }
    provenance_path = output.with_name(f"{output.stem}_merge_provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    provenance_path.chmod(0o600)
    print(json.dumps({"output": str(output), **provenance}, indent=2))


if __name__ == "__main__":
    main()
