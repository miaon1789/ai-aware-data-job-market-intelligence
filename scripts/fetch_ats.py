#!/usr/bin/env python3
"""Collect full-text Australian job ads from public company ATS boards.

For personal job-search analysis. Uses public, unauthenticated Greenhouse/Ashby/
Lever boards (no API key). Row-level records are written under data/private.

Add your own targets with --company provider:token[:Display Name], e.g.
  --company greenhouse:canva --company ashby:SomeStartup:"Some Startup"
The token is the slug in the company's careers URL.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_market_intelligence.ats import (  # noqa: E402
    DEFAULT_COMPANIES,
    PROVIDERS,
    ATSClient,
    collect_ats_jobs,
)

PRIVATE_ROOT = ROOT / "data/private"


def private_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def validate_private_output(path: Path) -> Path:
    resolved = path.resolve()
    if PRIVATE_ROOT.resolve() not in resolved.parents:
        raise ValueError("row-level ATS output must be stored under data/private")
    return resolved


def parse_company(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":", 2)
    provider = parts[0].strip().lower()
    if provider not in PROVIDERS or len(parts) < 2 or not parts[1].strip():
        raise argparse.ArgumentTypeError(
            f"--company must be provider:token[:display] with provider in {', '.join(PROVIDERS)}"
        )
    token = parts[1].strip()
    display = parts[2].strip() if len(parts) == 3 and parts[2].strip() else token
    return provider, token, display


def safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:60] or "board"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--company",
        action="append",
        type=parse_company,
        dest="companies",
        help="provider:token[:display]; repeatable. Defaults to the verified seed list.",
    )
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "data/private/ats/job_ads.csv")
    parser.add_argument("--save-raw-responses", action="store_true")
    args = parser.parse_args()

    try:
        output_path = validate_private_output(args.output)
    except ValueError as exc:
        parser.error(str(exc))

    companies = args.companies or list(DEFAULT_COMPANIES)
    raw_directory = output_path.parent / "raw_responses"

    def save_raw(provider: str, token: str, payload: object) -> None:
        if not args.save_raw_responses:
            return
        path = raw_directory / f"{provider}_{safe_slug(token)}.json"
        private_write(path, json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"Fetching {len(companies)} company boards...", file=sys.stderr)
    result = collect_ats_jobs(
        ATSClient(),
        companies=companies,
        request_delay_seconds=args.delay_seconds,
        max_retries=args.max_retries,
        on_board=save_raw,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.chmod(0o700)
    result.jobs.to_csv(output_path, index=False)
    output_path.chmod(0o600)

    provenance = {
        "source": "Public company ATS boards (Greenhouse / Ashby / Lever)",
        "purpose": "non-commercial personal job-search research",
        "collected_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "companies": [f"{p}:{t}" for p, t, _ in (tuple(c) for c in companies)],
        "companies_fetched": result.companies_fetched,
        "companies_failed": [f"{p}:{t}" for p, t in result.companies_failed],
        "received_records": result.received_records,
        "kept_au_records": result.kept_records,
        "raw_responses_retained": args.save_raw_responses,
    }
    provenance_path = output_path.with_name(f"{output_path.stem}_provenance.json")
    private_write(provenance_path, json.dumps(provenance, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "private_output": str(output_path),
                "companies_fetched": result.companies_fetched,
                "companies_failed": [f"{p}:{t}" for p, t in result.companies_failed],
                "received_records": result.received_records,
                "kept_au_records": result.kept_records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
