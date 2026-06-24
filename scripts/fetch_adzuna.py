#!/usr/bin/env python3
"""Collect a private, bounded snapshot from the Adzuna Australia API."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_market_intelligence.adzuna import (  # noqa: E402
    ATTRIBUTION_URL,
    DEFAULT_CITIES,
    DEFAULT_QUERY_GROUPS,
    TERMS_URL,
    AdzunaClient,
    collect_job_ads,
)

PRIVATE_ROOT = ROOT / "data/private"


def load_env_file(path: Path) -> None:
    """Load only the two expected secrets without overriding the shell."""

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        if name in {"ADZUNA_APP_ID", "ADZUNA_APP_KEY"}:
            os.environ.setdefault(name, value.strip().strip("\"'"))


def private_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def validate_private_output(path: Path) -> Path:
    resolved = path.resolve()
    private_root = PRIVATE_ROOT.resolve()
    if private_root not in resolved.parents:
        raise ValueError("row-level Adzuna output must be stored under data/private")
    return resolved


def safe_query_slug(query: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:60] or "query"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument(
        "--query-group",
        action="append",
        choices=sorted(DEFAULT_QUERY_GROUPS),
        dest="query_groups",
        help=(
            "Named query group to collect; repeatable. Defaults to entry_level "
            "and ai_signal."
        ),
    )
    parser.add_argument("--city", action="append", choices=DEFAULT_CITIES, dest="cities")
    parser.add_argument("--pages-per-query", type=int, default=1)
    parser.add_argument("--results-per-page", type=int, default=50)
    parser.add_argument("--delay-seconds", type=float, default=2.5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/private/adzuna/job_ads.csv"
    )
    parser.add_argument(
        "--save-raw-responses",
        action="store_true",
        help="Privately retain full JSON pages; disabled by default",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    app_id = os.environ.get("ADZUNA_APP_ID", "")
    app_key = os.environ.get("ADZUNA_APP_KEY", "")
    if not app_id or not app_key:
        parser.error("set ADZUNA_APP_ID and ADZUNA_APP_KEY in the environment or .env")

    try:
        output_path = validate_private_output(args.output)
    except ValueError as exc:
        parser.error(str(exc))

    raw_directory = output_path.parent / "raw_responses"

    def save_raw(call_number: int, query: str, city: str, payload: dict) -> None:
        if not args.save_raw_responses:
            return
        safe_query = safe_query_slug(query)
        path = raw_directory / f"{call_number:03d}_{city.lower()}_{safe_query}.json"
        private_write(path, json.dumps(payload, ensure_ascii=False, indent=2))

    selected_groups = args.query_groups or ["entry_level", "ai_signal"]
    query_plan = {
        group: list(DEFAULT_QUERY_GROUPS[group])
        for group in selected_groups
    }
    if args.queries:
        query_plan["custom"] = args.queries
    cities = args.cities or list(DEFAULT_CITIES)
    result = collect_job_ads(
        AdzunaClient(app_id, app_key),
        queries=query_plan,
        cities=cities,
        pages_per_query=args.pages_per_query,
        results_per_page=args.results_per_page,
        request_delay_seconds=args.delay_seconds,
        max_retries=args.max_retries,
        on_raw_page=save_raw,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.chmod(0o700)
    result.jobs.to_csv(output_path, index=False)
    output_path.chmod(0o600)

    provenance = {
        "source": "The Adzuna API",
        "source_url": ATTRIBUTION_URL,
        "terms_url": TERMS_URL,
        "purpose": "non-commercial personal research",
        "collected_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "query_groups": selected_groups,
        "query_plan": query_plan,
        "cities": cities,
        "pages_per_query": args.pages_per_query,
        "api_calls": result.calls,
        "max_retries": args.max_retries,
        "received_records": result.received_records,
        "unique_records": len(result.jobs),
        "skipped_records": result.skipped_records,
        "raw_responses_retained": args.save_raw_responses,
        "public_release_policy": "aggregates only; no row-level job advertisements",
    }
    provenance_path = output_path.with_name(f"{output_path.stem}_provenance.json")
    private_write(
        provenance_path,
        json.dumps(provenance, ensure_ascii=False, indent=2),
    )
    print(
        json.dumps(
            {
                "private_output": str(output_path),
                "private_provenance": str(provenance_path),
                "unique_records": len(result.jobs),
                "api_calls": result.calls,
                "raw_responses_retained": args.save_raw_responses,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
