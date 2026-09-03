"""Collect full-text Australian job ads from public company ATS boards.

Greenhouse, Ashby and Lever expose public, unauthenticated JSON job boards that
return complete job descriptions (unlike the truncated Adzuna free tier). This
collector fetches configured companies, keeps only Australian roles, and emits
the shared DATA_CARD schema for personal job-search analysis. No API key needed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .adzuna import html_to_text
from .locations import normalise_au_city

GREENHOUSE = "greenhouse"
ASHBY = "ashby"
LEVER = "lever"
PROVIDERS = (GREENHOUSE, ASHBY, LEVER)

# Verified 2026-09-03 by probing each public board and counting postings whose
# location normalises to an Australian city. Each entry is (provider,
# board_token, display_name). Extend with your own targets — the token is the
# slug in the company's careers URL (boards.greenhouse.io/<token>,
# jobs.ashbyhq.com/<token>, jobs.lever.co/<token>). Boards move between vendors,
# so a token that stops resolving is expected rather than an error.
DEFAULT_COMPANIES = (
    (GREENHOUSE, "block", "Block"),
    (GREENHOUSE, "eucalyptus", "Eucalyptus"),
    (GREENHOUSE, "databricks", "Databricks"),
    (GREENHOUSE, "robinhood", "Robinhood"),
    (GREENHOUSE, "stripe", "Stripe"),
    (GREENHOUSE, "datadog", "Datadog"),
    (GREENHOUSE, "okta", "Okta"),
    (GREENHOUSE, "workato", "Workato"),
    (GREENHOUSE, "figma", "Figma"),
    (GREENHOUSE, "gitlab", "GitLab"),
    (GREENHOUSE, "elastic", "Elastic"),
    (GREENHOUSE, "kaluza", "Kaluza"),
    (GREENHOUSE, "fivetran", "Fivetran"),
    (GREENHOUSE, "neo4j", "Neo4j"),
    (GREENHOUSE, "twilio", "Twilio"),
    (GREENHOUSE, "thoughtworks", "Thoughtworks"),
    (GREENHOUSE, "rubrik", "Rubrik"),
    (GREENHOUSE, "sigmacomputing", "Sigma Computing"),
    (GREENHOUSE, "grafanalabs", "Grafana Labs"),
    (GREENHOUSE, "amplitude", "Amplitude"),
    (GREENHOUSE, "quantium", "Quantium"),
    (GREENHOUSE, "affirm", "Affirm"),
    (GREENHOUSE, "cultureamp", "Culture Amp"),
    (GREENHOUSE, "mongodb", "MongoDB"),
    (ASHBY, "Airwallex", "Airwallex"),
    (ASHBY, "lorikeet", "Lorikeet"),
    (ASHBY, "relevanceai", "Relevance AI"),
    (ASHBY, "vanta", "Vanta"),
    (ASHBY, "notion", "Notion"),
    (LEVER, "brighte", "Brighte"),
    (LEVER, "immutable", "Immutable"),
)

MIN_REQUEST_DELAY_SECONDS = 0.5
MAX_COMPANIES_PER_RUN = 200
_USER_AGENT = "JobMarketIntelligenceResearch/0.1 personal-research"

_OUTPUT_COLUMNS = (
    "job_id",
    "title",
    "description",
    "description_structured",
    "city",
    "company",
    "posted_at",
    "source",
    "source_url",
    "role_label",
    "collection_query",
    "collection_group",
    "retrieved_at",
    "api_city",
    "is_remote",
    "employment_type",
    "department",
)


class ATSAPIError(RuntimeError):
    """Raised without embedding the full request URL in the message."""


def _default_request_json(url: str, timeout: float) -> object:
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.load(response)
    except HTTPError as exc:
        raise ATSAPIError(f"ATS board returned HTTP {exc.code}") from None
    except URLError:
        raise ATSAPIError("ATS board could not be reached") from None
    except json.JSONDecodeError:
        raise ATSAPIError("ATS board returned invalid JSON") from None


def board_url(provider: str, token: str) -> str:
    if not token.strip():
        raise ValueError("board token must not be empty")
    if provider == GREENHOUSE:
        return f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
    if provider == ASHBY:
        return f"https://api.ashbyhq.com/posting-api/job-board/{token}"
    if provider == LEVER:
        return f"https://api.lever.co/v0/postings/{token}?mode=json"
    raise ValueError(f"provider must be one of: {', '.join(PROVIDERS)}")


class ATSClient:
    """Fetches public ATS boards; the request seam is injectable for testing."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        request_json: Callable[[str, float], object] | None = None,
    ) -> None:
        self._timeout = timeout
        self._request_json = request_json or _default_request_json

    def fetch_board(self, provider: str, token: str) -> object:
        return self._request_json(board_url(provider, token), self._timeout)


def _raw_jobs(provider: str, payload: object) -> list[dict]:
    if provider in (GREENHOUSE, ASHBY):
        if isinstance(payload, dict) and isinstance(payload.get("jobs"), list):
            return [job for job in payload["jobs"] if isinstance(job, dict)]
        raise ATSAPIError("ATS board response is missing a jobs list")
    if provider == LEVER:
        if isinstance(payload, list):
            return [job for job in payload if isinstance(job, dict)]
        raise ATSAPIError("ATS board response is missing a jobs list")
    raise ValueError(f"provider must be one of: {', '.join(PROVIDERS)}")


def _parse_date(raw: object, fallback: str) -> str:
    text = str(raw or "").strip()
    if text:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            pass
    return fallback


def _record(
    *,
    job_id: str,
    title: str,
    description: str,
    description_structured: str,
    city: str,
    company: str,
    posted_at: str,
    source: str,
    source_url: str,
    retrieved_at: str,
    token: str,
    provider: str,
    api_city: str,
    is_remote: bool,
    employment_type: str,
    department: str,
) -> dict[str, str] | None:
    if not job_id or len(title) < 2 or len(description) < 20:
        return None
    return {
        "job_id": job_id,
        "title": title,
        "description": description,
        "description_structured": description_structured or description,
        "city": city,
        "company": company or token,
        "posted_at": posted_at,
        "source": source,
        "source_url": source_url,
        "role_label": "",
        "collection_query": token,
        "collection_group": provider,
        "retrieved_at": retrieved_at,
        "api_city": api_city,
        "is_remote": "yes" if is_remote else "no",
        "employment_type": employment_type,
        "department": department,
    }


def _normalise_greenhouse(item: dict, *, company: str, token: str, retrieved_at: str):
    location = item.get("location")
    raw_city = str(location.get("name", "")) if isinstance(location, dict) else ""
    city = normalise_au_city(raw_city, require_signal=True)
    if city is None:
        return None
    departments = item.get("departments") or []
    first_dept = departments[0] if departments and isinstance(departments[0], dict) else {}
    return _record(
        job_id=f"gh-{str(item.get('id', '')).strip()}",
        title=html_to_text(str(item.get("title", ""))),
        description=html_to_text(str(item.get("content", ""))),
        description_structured=html_to_text(
            str(item.get("content", "")), preserve_structure=True
        ),
        city=city,
        company=html_to_text(str(item.get("company_name", ""))) or company,
        posted_at=_parse_date(
            item.get("first_published") or item.get("updated_at"), retrieved_at[:10]
        ),
        source="greenhouse-ats",
        source_url=str(item.get("absolute_url", "")).strip(),
        retrieved_at=retrieved_at,
        token=token,
        provider=GREENHOUSE,
        api_city=raw_city,
        is_remote="remote" in raw_city.lower(),
        employment_type="",
        department=str(first_dept.get("name", "")),
    )


def _ashby_location(item: dict) -> str:
    primary = str(item.get("location", "") or "")
    if normalise_au_city(primary, require_signal=True):
        return primary
    for secondary in item.get("secondaryLocations", []) or []:
        value = secondary.get("location", "") if isinstance(secondary, dict) else str(secondary)
        if normalise_au_city(value, require_signal=True):
            return str(value)
    return primary


def _normalise_ashby(item: dict, *, company: str, token: str, retrieved_at: str):
    raw_city = _ashby_location(item)
    city = normalise_au_city(raw_city, require_signal=True)
    if city is None:
        return None
    description = html_to_text(
        str(item.get("descriptionPlain") or item.get("descriptionHtml") or "")
    )
    # The HTML variant keeps the heading and bullet markup that plain text drops.
    structured = html_to_text(
        str(item.get("descriptionHtml") or item.get("descriptionPlain") or ""),
        preserve_structure=True,
    )
    return _record(
        job_id=f"ashby-{str(item.get('id', '')).strip()}",
        title=html_to_text(str(item.get("title", ""))),
        description=description,
        description_structured=structured,
        city=city,
        company=company,
        posted_at=_parse_date(item.get("publishedAt"), retrieved_at[:10]),
        source="ashby-ats",
        source_url=str(item.get("jobUrl") or item.get("applyUrl") or "").strip(),
        retrieved_at=retrieved_at,
        token=token,
        provider=ASHBY,
        api_city=raw_city,
        is_remote=bool(item.get("isRemote")) or "remote" in raw_city.lower(),
        employment_type=str(item.get("employmentType", "")),
        department=str(item.get("department", "")),
    )


def _normalise_lever(item: dict, *, company: str, token: str, retrieved_at: str):
    categories = item.get("categories", {}) if isinstance(item.get("categories"), dict) else {}
    raw_city = str(categories.get("location", "") or "")
    city = normalise_au_city(raw_city, require_signal=True)
    if city is None:
        return None
    description = html_to_text(str(item.get("descriptionPlain") or item.get("description") or ""))
    structured = html_to_text(
        str(item.get("description") or item.get("descriptionPlain") or ""),
        preserve_structure=True,
    )
    return _record(
        job_id=f"lever-{str(item.get('id', '')).strip()}",
        title=html_to_text(str(item.get("text", ""))),
        description=description,
        description_structured=structured,
        city=city,
        company=company,
        posted_at=_parse_date(item.get("createdAt"), retrieved_at[:10]),
        source="lever-ats",
        source_url=str(item.get("hostedUrl") or item.get("applyUrl") or "").strip(),
        retrieved_at=retrieved_at,
        token=token,
        provider=LEVER,
        api_city=raw_city,
        is_remote="remote" in raw_city.lower(),
        employment_type=str(categories.get("commitment", "")),
        department=str(categories.get("team", "")),
    )


_NORMALISERS = {
    GREENHOUSE: _normalise_greenhouse,
    ASHBY: _normalise_ashby,
    LEVER: _normalise_lever,
}


@dataclass
class CollectionResult:
    jobs: pd.DataFrame
    companies_fetched: int
    companies_failed: list[tuple[str, str]]
    received_records: int
    kept_records: int


def _normalise_company(entry: Sequence[str]) -> tuple[str, str, str]:
    parts = list(entry)
    if len(parts) == 2:
        provider, token = parts
        display = token
    elif len(parts) == 3:
        provider, token, display = parts
    else:
        raise ValueError("company entries must be (provider, token) or (provider, token, display)")
    provider = str(provider).strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of: {', '.join(PROVIDERS)}")
    if not str(token).strip():
        raise ValueError("board token must not be empty")
    return provider, str(token).strip(), str(display).strip() or str(token).strip()


def collect_ats_jobs(
    client: ATSClient,
    *,
    companies: Sequence[Sequence[str]] = DEFAULT_COMPANIES,
    request_delay_seconds: float = 1.0,
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    on_board: Callable[[str, str, object], None] | None = None,
    skip_failures: bool = True,
) -> CollectionResult:
    """Fetch each company board, keep Australian roles, and deduplicate by id."""

    if request_delay_seconds < MIN_REQUEST_DELAY_SECONDS:
        raise ValueError(f"request_delay_seconds must be at least {MIN_REQUEST_DELAY_SECONDS}")
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    plan = [_normalise_company(entry) for entry in companies]
    if not plan:
        raise ValueError("at least one company is required")
    if len(plan) > MAX_COMPANIES_PER_RUN:
        raise ValueError(f"collection is limited to {MAX_COMPANIES_PER_RUN} companies per run")

    retrieved_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    records_by_id: dict[str, dict[str, str]] = {}
    received_records = 0
    companies_fetched = 0
    companies_failed: list[tuple[str, str]] = []

    for index, (provider, token, display) in enumerate(plan):
        if index:
            sleep(request_delay_seconds)
        payload: object | None = None
        for attempt in range(1, max_retries + 1):
            try:
                payload = client.fetch_board(provider, token)
                break
            except ATSAPIError:
                if attempt == max_retries:
                    if skip_failures:
                        companies_failed.append((provider, token))
                        payload = None
                        break
                    raise
                sleep(request_delay_seconds * attempt)
        if payload is None:
            continue
        companies_fetched += 1
        if on_board:
            on_board(provider, token, payload)
        normalise = _NORMALISERS[provider]
        for item in _raw_jobs(provider, payload):
            received_records += 1
            record = normalise(item, company=display, token=token, retrieved_at=retrieved_at)
            if record is None:
                continue
            records_by_id.setdefault(record["job_id"], record)

    jobs = pd.DataFrame(records_by_id.values(), columns=list(_OUTPUT_COLUMNS))
    if not jobs.empty:
        jobs = jobs.sort_values(["company", "job_id"]).reset_index(drop=True)
    return CollectionResult(
        jobs=jobs,
        companies_fetched=companies_fetched,
        companies_failed=companies_failed,
        received_records=received_records,
        kept_records=len(jobs),
    )
