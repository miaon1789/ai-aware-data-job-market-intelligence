"""Terms-aware client for personal job-search research with the JSearch API.

JSearch (RapidAPI) aggregates Google for Jobs results, so it returns complete
job descriptions rather than the truncated excerpts of the Adzuna free tier.
This collector is for private, personal job-search analysis: row-level records
stay under ``data/private`` and no network request is made without a key.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .adzuna import html_to_text

SEARCH_ENDPOINT = "https://jsearch.p.rapidapi.com/search"
RAPIDAPI_HOST = "jsearch.p.rapidapi.com"
ATTRIBUTION = "Aggregated from Google for Jobs via the JSearch API (RapidAPI)."

# Query groups tuned to a Computing graduate (Data Science / ML specialisation)
# who is applying to entry-level data, general IT/software and ML/AI roles.
DATA_QUERIES = (
    "graduate data analyst",
    "junior data analyst",
    "entry level data analyst",
    "graduate data scientist",
    "junior data scientist",
    "graduate data engineer",
    "business intelligence analyst",
)
IT_QUERIES = (
    "graduate software engineer",
    "junior software engineer",
    "graduate software developer",
    "junior developer",
    "entry level software developer",
    "IT graduate program",
    "graduate backend developer",
)
ML_QUERIES = (
    "graduate machine learning engineer",
    "junior machine learning engineer",
    "graduate AI engineer",
    "junior AI engineer",
    "machine learning graduate",
    "MLOps engineer",
    "AI ML graduate",
)
DEFAULT_QUERY_GROUPS = {
    "entry_data": DATA_QUERIES,
    "entry_it": IT_QUERIES,
    "entry_ml": ML_QUERIES,
}
DEFAULT_CITIES = ("Sydney", "Melbourne")
DEFAULT_COUNTRY = "au"
DATE_POSTED_CHOICES = ("all", "today", "3days", "week", "month")
MIN_REQUEST_DELAY_SECONDS = 1.0
MAX_PAGES_PER_QUERY = 20
MAX_CALLS_PER_RUN = 100
QueryPlan = Mapping[str, Sequence[str]] | Sequence[str]

# Extra columns kept in the private CSV for manual job-search browsing. They are
# ignored by the analysis pipeline (see ingestion.REQUIRED_COLUMNS) but let you
# eyeball salary/remote/type without opening every posting.
_EXTRA_COLUMNS = (
    "api_city",
    "is_remote",
    "employment_type",
    "salary_min",
    "salary_max",
    "salary_period",
)
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
    *_EXTRA_COLUMNS,
)


class JSearchAPIError(RuntimeError):
    """Raised without embedding the API key or full request URL in the message."""


def _default_request_json(
    url: str, headers: dict[str, str], timeout: float
) -> dict[str, object]:
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.load(response)
    except HTTPError as exc:
        raise JSearchAPIError(f"JSearch API returned HTTP {exc.code}") from None
    except URLError:
        raise JSearchAPIError("JSearch API could not be reached") from None
    except json.JSONDecodeError:
        raise JSearchAPIError("JSearch API returned invalid JSON") from None


class JSearchClient:
    """Small API client that never logs the request URL or the RapidAPI key."""

    def __init__(
        self,
        api_key: str,
        *,
        host: str = RAPIDAPI_HOST,
        timeout: float = 30.0,
        request_json: Callable[[str, dict[str, str], float], dict[str, object]] | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("JSEARCH_API_KEY is required")
        self._api_key = api_key
        self._host = host
        self._timeout = timeout
        self._request_json = request_json or _default_request_json

    def _headers(self) -> dict[str, str]:
        return {
            "X-RapidAPI-Key": self._api_key,
            "X-RapidAPI-Host": self._host,
            "User-Agent": "JobMarketIntelligenceResearch/0.1 personal-research",
        }

    def search(
        self,
        *,
        query: str,
        location: str,
        page: int = 1,
        date_posted: str = "month",
        country: str = DEFAULT_COUNTRY,
    ) -> dict[str, object]:
        if page < 1:
            raise ValueError("page must be at least 1")
        if not query.strip():
            raise ValueError("query must not be empty")
        if not location.strip():
            raise ValueError("location must not be empty")
        if date_posted not in DATE_POSTED_CHOICES:
            raise ValueError(f"date_posted must be one of: {', '.join(DATE_POSTED_CHOICES)}")

        parameters = urlencode(
            {
                "query": f"{query.strip()} in {location.strip()}",
                "page": page,
                "num_pages": 1,
                "country": country,
                "date_posted": date_posted,
            }
        )
        payload = self._request_json(
            f"{SEARCH_ENDPOINT}?{parameters}", self._headers(), self._timeout
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise JSearchAPIError("JSearch API response is missing a data list")
        return payload


@dataclass
class CollectionResult:
    jobs: pd.DataFrame
    calls: int
    received_records: int
    skipped_records: int


def _parse_posted_date(item: dict, fallback: str) -> str:
    raw = str(item.get("job_posted_at_datetime_utc") or "").strip()
    if raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            pass
    timestamp = item.get("job_posted_at_timestamp")
    if isinstance(timestamp, (int, float)) and timestamp > 0:
        try:
            return datetime.fromtimestamp(timestamp, tz=UTC).date().isoformat()
        except (ValueError, OverflowError, OSError):
            pass
    return fallback


def _optional_str(value: object) -> str:
    if value in (None, ""):
        return ""
    return str(value).strip()


def _normalise_job(
    item: object,
    *,
    query: str,
    query_group: str,
    location: str,
    retrieved_at: str,
) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None
    job_id = str(item.get("job_id", "")).strip()
    title = html_to_text(str(item.get("job_title", "")))
    description = html_to_text(str(item.get("job_description", "")))
    if not job_id or len(title) < 2 or len(description) < 20:
        return None

    company = html_to_text(str(item.get("employer_name", ""))) or "Undisclosed"
    source_url = _optional_str(item.get("job_apply_link")) or _optional_str(
        item.get("job_google_link")
    )

    return {
        "job_id": job_id,
        "title": title,
        "description": description,
        "description_structured": html_to_text(
            str(item.get("job_description", "")), preserve_structure=True
        ),
        # Canonical city is the searched region so it satisfies the pipeline's
        # JobAd.SUPPORTED_CITIES check; the precise API city goes to api_city.
        "city": location,
        "company": company,
        "posted_at": _parse_posted_date(item, retrieved_at[:10]),
        "source": "jsearch-api",
        "source_url": source_url,
        "role_label": "",
        "collection_query": query,
        "collection_group": query_group,
        "retrieved_at": retrieved_at,
        "api_city": html_to_text(str(item.get("job_city") or "")),
        "is_remote": "yes" if item.get("job_is_remote") else "no",
        "employment_type": _optional_str(item.get("job_employment_type")),
        "salary_min": _optional_str(item.get("job_min_salary")),
        "salary_max": _optional_str(item.get("job_max_salary")),
        "salary_period": _optional_str(item.get("job_salary_period")),
    }


def _expand_query_plan(queries: QueryPlan) -> list[tuple[str, str]]:
    if isinstance(queries, Mapping):
        plan = [
            (str(group), str(query))
            for group, group_queries in queries.items()
            for query in group_queries
        ]
    else:
        plan = [("custom", str(query)) for query in queries]
    cleaned = [(group.strip(), query.strip()) for group, query in plan]
    invalid = [(group, query) for group, query in cleaned if not group or not query]
    if invalid:
        raise ValueError("query groups and queries must not be empty")
    return cleaned


def _merge_collection_provenance(existing: dict[str, str], query: str, query_group: str) -> None:
    terms = set(existing["collection_query"].split(" | "))
    terms.add(query)
    existing["collection_query"] = " | ".join(sorted(terms))
    groups = set(existing["collection_group"].split(" | "))
    groups.add(query_group)
    existing["collection_group"] = " | ".join(sorted(groups))


def collect_job_ads(
    client: JSearchClient,
    *,
    queries: QueryPlan = DEFAULT_QUERY_GROUPS,
    cities: Sequence[str] = DEFAULT_CITIES,
    pages_per_query: int = 1,
    date_posted: str = "month",
    country: str = DEFAULT_COUNTRY,
    request_delay_seconds: float = MIN_REQUEST_DELAY_SECONDS,
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    on_raw_page: Callable[[int, str, str, dict[str, object]], None] | None = None,
) -> CollectionResult:
    """Collect a bounded, rate-limited snapshot and deduplicate it by job id."""

    if not 1 <= pages_per_query <= MAX_PAGES_PER_QUERY:
        raise ValueError(f"pages_per_query must be between 1 and {MAX_PAGES_PER_QUERY}")
    if request_delay_seconds < MIN_REQUEST_DELAY_SECONDS:
        raise ValueError(
            f"request_delay_seconds must be at least {MIN_REQUEST_DELAY_SECONDS}"
        )
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    locations = [str(city).strip() for city in cities]
    if not locations or any(not city for city in locations):
        raise ValueError("cities must be a non-empty list of location names")
    query_plan = _expand_query_plan(queries)
    calls_planned = len(query_plan) * len(locations) * pages_per_query
    if calls_planned > MAX_CALLS_PER_RUN:
        raise ValueError(f"collection is limited to {MAX_CALLS_PER_RUN} API calls per run")

    retrieved_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    records_by_id: dict[str, dict[str, str]] = {}
    received_records = 0
    skipped_records = 0
    call_number = 0

    for location in locations:
        for query_group, query in query_plan:
            for page in range(1, pages_per_query + 1):
                if call_number:
                    sleep(request_delay_seconds)
                for attempt in range(1, max_retries + 1):
                    try:
                        payload = client.search(
                            query=query,
                            location=location,
                            page=page,
                            date_posted=date_posted,
                            country=country,
                        )
                        break
                    except JSearchAPIError:
                        if attempt == max_retries:
                            raise
                        sleep(request_delay_seconds * attempt)
                call_number += 1
                if on_raw_page:
                    on_raw_page(call_number, query, location, payload)
                for item in payload["data"]:
                    received_records += 1
                    record = _normalise_job(
                        item,
                        query=query,
                        query_group=query_group,
                        location=location,
                        retrieved_at=retrieved_at,
                    )
                    if record is None:
                        skipped_records += 1
                        continue
                    existing = records_by_id.get(record["job_id"])
                    if existing:
                        _merge_collection_provenance(existing, query, query_group)
                    else:
                        records_by_id[record["job_id"]] = record

    jobs = pd.DataFrame(records_by_id.values(), columns=list(_OUTPUT_COLUMNS))
    if not jobs.empty:
        jobs = jobs.sort_values("job_id").reset_index(drop=True)
    return CollectionResult(
        jobs=jobs,
        calls=call_number,
        received_records=received_records,
        skipped_records=skipped_records,
    )
