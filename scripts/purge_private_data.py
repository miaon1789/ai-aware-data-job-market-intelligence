#!/usr/bin/env python3
"""Review or delete expired private data.

Covers two stores, because private advertisement text lives in both. The
filesystem under `data/private/` is the obvious one. The other is the Docker
volume backing the Doccano annotation server: annotating in Doccano copies the
advertisement text into its database, where it survives deleting the repository
and is invisible to anything that walks the working tree. `docs/ADZUNA_USAGE.md`
promises a complete purge on API termination, and that promise is not kept by
a filesystem sweep alone.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIVATE_ROOT = ROOT / "data/private"

# Docker volumes that hold advertisement text copied out of data/private.
# Doccano stores every annotated document in its own SQLite database.
PRIVATE_DOCKER_VOLUMES = ("doccano-db",)
PRIVATE_DOCKER_CONTAINERS = ("doccano",)


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


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(  # noqa: S603
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def existing_private_volumes() -> list[str]:
    """Names of known private Docker volumes that are actually present."""

    if not _docker_available():
        return []
    listed = subprocess.run(  # noqa: S603
        ["docker", "volume", "ls", "--format", "{{.Name}}"],
        capture_output=True,
        text=True,
    )
    present = set(listed.stdout.split())
    return [name for name in PRIVATE_DOCKER_VOLUMES if name in present]


def remove_private_volumes(volumes: list[str]) -> list[str]:
    """Remove the named volumes and any container still holding them."""

    removed: list[str] = []
    for container in PRIVATE_DOCKER_CONTAINERS:
        subprocess.run(  # noqa: S603
            ["docker", "rm", "-f", container], capture_output=True, text=True
        )
    for volume in volumes:
        result = subprocess.run(  # noqa: S603
            ["docker", "volume", "rm", volume], capture_output=True, text=True
        )
        if result.returncode == 0:
            removed.append(volume)
    return removed


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
    parser.add_argument(
        "--include-docker-volumes",
        action="store_true",
        help=(
            "Also remove Docker volumes holding annotated advertisement text "
            "(the Doccano database). Required for the complete purge described "
            "in docs/ADZUNA_USAGE.md."
        ),
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
    private_volumes = existing_private_volumes()
    removed_volumes: list[str] = []
    deleted = 0
    if args.confirm and args.include_docker_volumes:
        removed_volumes = remove_private_volumes(private_volumes)
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
                "private_docker_volumes_present": private_volumes,
                "private_docker_volumes_removed": removed_volumes,
                "docker_volume_note": (
                    ""
                    if not private_volumes or args.include_docker_volumes
                    else (
                        "Advertisement text is also stored in the Docker volumes "
                        "listed above. A filesystem purge does not remove it; "
                        "pass --include-docker-volumes to include them."
                    )
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
