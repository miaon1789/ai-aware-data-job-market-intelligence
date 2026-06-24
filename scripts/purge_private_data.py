#!/usr/bin/env python3
"""Review or delete expired private data under data/private."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIVATE_ROOT = ROOT / "data/private"


def expired_files(
    private_root: Path,
    *,
    retention_days: int,
    purge_all: bool = False,
    now: float | None = None,
) -> list[Path]:
    if retention_days < 1:
        raise ValueError("retention_days must be at least 1")
    if not private_root.exists():
        return []
    current_time = now if now is not None else time.time()
    cutoff = current_time - retention_days * 24 * 60 * 60
    return sorted(
        path
        for path in private_root.rglob("*")
        if path.is_file() and not path.is_symlink() and (purge_all or path.stat().st_mtime < cutoff)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--retention-days", type=int, default=90)
    parser.add_argument(
        "--purge-all",
        action="store_true",
        help="Select every private file, for example after API access termination",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete selected files; without this flag the command is a dry run",
    )
    args = parser.parse_args()

    private_root = args.private_root.resolve()
    allowed_root = DEFAULT_PRIVATE_ROOT.resolve()
    if private_root != allowed_root and allowed_root not in private_root.parents:
        parser.error("private-root must be data/private or one of its subdirectories")

    selected = expired_files(
        private_root,
        retention_days=args.retention_days,
        purge_all=args.purge_all,
    )
    deleted = 0
    if args.confirm:
        for path in selected:
            path.unlink()
            deleted += 1
        for directory in sorted(
            (path for path in private_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass

    print(
        json.dumps(
            {
                "mode": "delete" if args.confirm else "dry-run",
                "private_root": str(private_root),
                "retention_days": args.retention_days,
                "purge_all": args.purge_all,
                "selected_files": len(selected),
                "deleted_files": deleted,
                "files": [str(path.relative_to(private_root)) for path in selected],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
