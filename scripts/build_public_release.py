#!/usr/bin/env python3
"""Create a reviewable aggregate-only release from the private DuckDB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_market_intelligence.publishing import (  # noqa: E402
    MINIMUM_REAL_CELL_SIZE,
    release_from_database,
    write_public_release,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/processed/job_market.duckdb"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/public")
    parser.add_argument(
        "--minimum-cell-size", type=int, default=MINIMUM_REAL_CELL_SIZE
    )
    parser.add_argument(
        "--document-labels",
        type=Path,
        help=(
            "Optional private adjudicated document_labels.csv. When provided, "
            "the public release is limited to the labelled semantic sample and "
            "uses role_label as the published role family."
        ),
    )
    args = parser.parse_args()

    if not args.database.exists():
        parser.error("private database does not exist; run the analysis pipeline first")
    if args.document_labels and not args.document_labels.is_file():
        parser.error(f"document labels do not exist: {args.document_labels}")
    release = release_from_database(
        args.database,
        minimum_cell_size=args.minimum_cell_size,
        document_labels_path=args.document_labels,
    )
    write_public_release(release, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "data_kind": release.metadata["data_kind"],
                "minimum_cell_size": release.metadata["minimum_cell_size"],
                "table_rows": {
                    name: len(frame) for name, frame in release.tables.items()
                },
                "contains_row_level_records": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
