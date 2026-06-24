"""Terms-aware client for personal research with the Adzuna Australia API."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

SEARCH_ENDPOINT = "https://api.adzuna.com/v1/api/jobs/au/search/{page}"
TERMS_URL = "https://developer.adzuna.com/docs/terms_of_service"
ATTRIBUTION_URL = "https://www.adzuna.com.au/"
DEFAULT_QUERIES = (
    "data analyst",
    "business intelligence analyst",
    "data scientist",
    "machine learning engineer",
    "data engineer",
    "analytics engineer",
)
ENTRY_LEVEL_QUERIES = (
    "graduate analyst",
    "junior analyst",
    "entry level analyst",
    "graduate data analyst",
    "junior data analyst",
    "entry level data analyst",
    "graduate business analyst",
    "junior business analyst",
    "graduate BI analyst",
    "junior BI analyst",
    "analytics graduate",
    "data analytics graduate",
    "analytics intern",
    "data internship",
    "data intern",
    "data analytics intern",
    "graduate data engineer",
    "junior data engineer",
    "graduate data scientist",
    "junior data scientist",
    "junior business intelligence analyst",
    "business intelligence internship",
)
AI_SIGNAL_QUERIES = (
    "AI analyst",
    "generative AI analyst",
    "LLM analyst",
    "automation analyst",
    "AI engineer graduate",
    "junior AI engineer",
    "prompt engineer",
    "data analyst ChatGPT",
    "data analyst Copilot",
)
DEFAULT_QUERY_GROUPS = {
    "entry_level": ENTRY_LEVEL_QUERIES,
    "ai_signal": AI_SIGNAL_QUERIES,
    "general_baseline": DEFAULT_QUERIES,
}
DEFAULT_CITIES = ("Sydney", "Melbourne")
MIN_REQUEST_DELAY_SECONDS = 2.4
MAX_RESULTS_PER_PAGE = 50
MAX_CALLS_PER_RUN = 100
QueryPlan = Mapping[str, Sequence[str]] | Sequence[str]


class AdzunaAPIError(RuntimeError):
    """Raised without embedding credentials or request URLs in the message."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style"}:
            self.hidden_depth += 1
        elif tag.lower() in {"p", "br", "li", "div"} and self.parts:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self.hidden_depth:
            self.hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)


def html_to_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(unescape(str(value)))
    return " ".join("".join(parser.parts).split())


def _default_request_json(url: str, timeout: float) -> dict[str, object]:
    request = Request(
        url,
        headers={"User-Agent": "JobMarketIntelligenceResearch/0.1 personal-research"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.load(response)
    except HTTPError as exc:
        raise AdzunaAPIError(f"Adzuna API returned HTTP {exc.code}") from None
    except URLError:
        raise AdzunaAPIError("Adzuna API could not be reached") from None
    except json.JSONDecodeError:
        raise AdzunaAPIError("Adzuna API returned invalid JSON") from None


class AdzunaClient:
    """Small API client that never logs request URLs or credentials."""

    def __init__(
        self,
        app_id: str,
        app_key: str,
        *,
        timeout: float = 30.0,
        request_json: Callable[[str, float], dict[str, object]] | None = None,
    ) -> None:
        if not app_id.strip() or not app_key.strip():
            raise ValueError("ADZUNA_APP_ID and ADZUNA_APP_KEY are required")
        self._app_id = app_id
        self._app_key = app_key
        self._timeout = timeout
        self._request_json = request_json or _default_request_json

    def search(
        self,
        *,
        query: str,
        city: str,
        page: int,
        results_per_page: int = MAX_RESULTS_PER_PAGE,
    ) -> dict[str, object]:
        if page < 1:
            raise ValueError("page must be at least 1")
        if not 1 <= results_per_page <= MAX_RESULTS_PER_PAGE:
            raise ValueError(f"results_per_page must be between 1 and {MAX_RESULTS_PER_PAGE}")
        if city not in DEFAULT_CITIES:
            raise ValueError(f"city must be one of: {', '.join(DEFAULT_CITIES)}")
        if not query.strip():
            raise ValueError("query must not be empty")

        parameters = urlencode(
            {
                "app_id": self._app_id,
                "app_key": self._app_key,
                "results_per_page": results_per_page,
                "what": query,
                "where": city,
                "sort_by": "date",
                "content-type": "application/json",
            }
        )
        payload = self._request_json(
            f"{SEARCH_ENDPOINT.format(page=page)}?{parameters}", self._timeout
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise AdzunaAPIError("Adzuna API response is missing a results list")
        return payload


@dataclass
class CollectionResult:
    jobs: pd.DataFrame
    calls: int
    received_records: int
    skipped_records: int


def _normalise_ad(
    item: object,
    *,
    query: str,
    query_group: str,
    city: str,
    retrieved_at: str,
) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None
    job_id = str(item.get("id", "")).strip()
    title = html_to_text(str(item.get("title", "")))
    description = html_to_text(str(item.get("description", "")))
    if not job_id or len(title) < 2 or len(description) < 20:
        return None

    company_value = item.get("company")
    company = "Undisclosed"
    if isinstance(company_value, dict):
        company = html_to_text(str(company_value.get("display_name", ""))) or company

    created = str(item.get("created", ""))
    try:
        posted_at = datetime.fromisoformat(created.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        posted_at = retrieved_at[:10]

    return {
        "job_id": job_id,
        "title": title,
        "description": description,
        "city": city,
        "company": company,
        "posted_at": posted_at,
        "source": "adzuna-api",
        "source_url": str(item.get("redirect_url", "")).strip(),
        "role_label": "",
        "collection_query": query,
        "collection_group": query_group,
        "retrieved_at": retrieved_at,
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


def collect_job_ads(
    client: AdzunaClient,
    *,
    queries: QueryPlan = DEFAULT_QUERIES,
    cities: list[str] | tuple[str, ...] = DEFAULT_CITIES,
    pages_per_query: int = 1,
    results_per_page: int = MAX_RESULTS_PER_PAGE,
    request_delay_seconds: float = MIN_REQUEST_DELAY_SECONDS,
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    on_raw_page: Callable[[int, str, str, dict[str, object]], None] | None = None,
) -> CollectionResult:
    """Collect a bounded, rate-limited snapshot and deduplicate it by Adzuna ID."""

    if not 1 <= pages_per_query <= 5:
        raise ValueError("pages_per_query must be between 1 and 5")
    if request_delay_seconds < MIN_REQUEST_DELAY_SECONDS:
        raise ValueError(
            f"request_delay_seconds must be at least {MIN_REQUEST_DELAY_SECONDS}"
        )
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    query_plan = _expand_query_plan(queries)
    calls_planned = len(query_plan) * len(cities) * pages_per_query
    if calls_planned > MAX_CALLS_PER_RUN:
        raise ValueError(f"collection is limited to {MAX_CALLS_PER_RUN} API calls per run")

    retrieved_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    records_by_id: dict[str, dict[str, str]] = {}
    received_records = 0
    skipped_records = 0
    call_number = 0

    for city in cities:
        for query_group, query in query_plan:
            for page in range(1, pages_per_query + 1):
                if call_number:
                    sleep(request_delay_seconds)
                for attempt in range(1, max_retries + 1):
                    try:
                        payload = client.search(
                            query=query,
                            city=city,
                            page=page,
                            results_per_page=results_per_page,
                        )
                        break
                    except AdzunaAPIError:
                        if attempt == max_retries:
                            raise
                        sleep(request_delay_seconds * attempt)
                call_number += 1
                if on_raw_page:
                    on_raw_page(call_number, query, city, payload)
                for item in payload["results"]:
                    received_records += 1
                    record = _normalise_ad(
                        item,
                        query=query,
                        query_group=query_group,
                        city=city,
                        retrieved_at=retrieved_at,
                    )
                    if record is None:
                        skipped_records += 1
                        continue
                    existing = records_by_id.get(record["job_id"])
                    if existing:
                        terms = set(existing["collection_query"].split(" | "))
                        terms.add(query)
                        existing["collection_query"] = " | ".join(sorted(terms))
                        groups = set(existing["collection_group"].split(" | "))
                        groups.add(query_group)
                        existing["collection_group"] = " | ".join(sorted(groups))
                    else:
                        records_by_id[record["job_id"]] = record

    columns = [
        "job_id",
        "title",
        "description",
        "city",
        "company",
        "posted_at",
        "source",
        "source_url",
        "role_label",
        "collection_query",
        "collection_group",
        "retrieved_at",
    ]
    jobs = pd.DataFrame(records_by_id.values(), columns=columns)
    if not jobs.empty:
        jobs = jobs.sort_values("job_id").reset_index(drop=True)
    return CollectionResult(
        jobs=jobs,
        calls=call_number,
        received_records=received_records,
        skipped_records=skipped_records,
    )
