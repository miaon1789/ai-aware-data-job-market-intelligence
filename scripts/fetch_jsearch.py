#!/usr/bin/env python3
"""Collect a private, bounded snapshot from the JSearch (Google for Jobs) API.

For personal job-search analysis only. Row-level records are written under
data/private (Git-ignored). No network request is made until JSEARCH_API_KEY is
configured. Get a free key at https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch
"""

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

from job_market_intelligence.jsearch import (  # noqa: E402
    ATTRIBUTION,
    DATE_POSTED_CHOICES,
    DEFAULT_CITIES,
    DEFAULT_COUNTRY,
    DEFAULT_QUERY_GROUPS,
    JSearchClient,
    collect_job_ads,
)

PRIVATE_ROOT = ROOT / "data/private"


def load_env_file(path: Path) -> None:
    """Load only the expected secret without overriding the shell."""

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        if name == "JSEARCH_API_KEY":
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
        raise ValueError("row-level JSearch output must be stored under data/private")
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
        help="Named query group to collect; repeatable. Defaults to all three groups.",
    )
    parser.add_argument(
        "--city",
        action="append",
        dest="cities",
        help="Location to search; repeatable. Defaults to Sydney and Melbourne.",
    )
    parser.add_argument("--pages-per-query", type=int, default=1)
    parser.add_argument(
        "--date-posted", choices=DATE_POSTED_CHOICES, default="month"
    )
    parser.add_argument("--country", default=DEFAULT_COUNTRY)
    parser.add_argument("--delay-seconds", type=float, default=1.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/private/jsearch/job_ads.csv"
    )
    parser.add_argument(
        "--save-raw-responses",
        action="store_true",
        help="Privately retain full JSON pages; disabled by default",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    api_key = os.environ.get("JSEARCH_API_KEY", "")
    if not api_key:
        parser.error(
            "set JSEARCH_API_KEY in the environment or .env "
            "(free key: https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch)"
        )

    try:
        output_path = validate_private_output(args.output)
    except ValueError as exc:
        parser.error(str(exc))

    raw_directory = output_path.parent / "raw_responses"

    def save_raw(call_number: int, query: str, location: str, payload: dict) -> None:
        if not args.save_raw_responses:
            return
        safe_query = safe_query_slug(query)
        path = raw_directory / f"{call_number:03d}_{safe_query_slug(location)}_{safe_query}.json"
        private_write(path, json.dumps(payload, ensure_ascii=False, indent=2))

    # Explicit --query alone means "only these custom queries" (cheap, precise).
    # Only fall back to all default groups when neither queries nor groups are given.
    if args.query_groups:
        selected_groups = args.query_groups
    elif args.queries:
        selected_groups = []
    else:
        selected_groups = list(DEFAULT_QUERY_GROUPS)
    query_plan = {group: list(DEFAULT_QUERY_GROUPS[group]) for group in selected_groups}
    if args.queries:
        query_plan["custom"] = args.queries
    cities = args.cities or list(DEFAULT_CITIES)

    planned_calls = sum(len(v) for v in query_plan.values()) * len(cities) * args.pages_per_query
    print(f"Planned API calls: {planned_calls} (mind your monthly RapidAPI quota)", file=sys.stderr)

    result = collect_job_ads(
        JSearchClient(api_key),
        queries=query_plan,
        cities=cities,
        pages_per_query=args.pages_per_query,
        date_posted=args.date_posted,
        country=args.country,
        request_delay_seconds=args.delay_seconds,
        max_retries=args.max_retries,
        on_raw_page=save_raw,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.chmod(0o700)
    result.jobs.to_csv(output_path, index=False)
    output_path.chmod(0o600)

    provenance = {
        "source": "JSearch API (RapidAPI)",
        "attribution": ATTRIBUTION,
        "purpose": "non-commercial personal job-search research",
        "collected_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "query_groups": selected_groups,
        "query_plan": query_plan,
        "cities": cities,
        "date_posted": args.date_posted,
        "country": args.country,
        "pages_per_query": args.pages_per_query,
        "api_calls": result.calls,
        "max_retries": args.max_retries,
        "received_records": result.received_records,
        "unique_records": len(result.jobs),
        "skipped_records": result.skipped_records,
        "raw_responses_retained": args.save_raw_responses,
    }
    provenance_path = output_path.with_name(f"{output_path.stem}_provenance.json")
    private_write(provenance_path, json.dumps(provenance, ensure_ascii=False, indent=2))
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
