#!/usr/bin/env python3
"""MCP server exposing the job-advertisement corpus as two agent tools.

The two tools exist because they answer different shapes of question, and the
tool descriptions below are the mechanism that gets a model to pick correctly.
This is the part of the system most easily got wrong: a single "search the job
ads" tool will be handed "how many ads require SQL?", will return five
passages, and the model will confidently invent a number from them.

So `search_job_ads` says plainly what it cannot do, and `query_job_stats` says
plainly when it must be preferred. The descriptions are written for the model
that reads them, not for a human skimming a README.

Runs over stdio and reads a local private index; it is not a network service.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from job_market_intelligence.rag.routing import classify_query, routing_rationale  # noqa: E402
from job_market_intelligence.rag.stats import DIMENSIONS, JobStatsService  # noqa: E402
from job_market_intelligence.retrieval.config import RetrievalConfig  # noqa: E402
from job_market_intelligence.retrieval.embedding import ChunkEncoder  # noqa: E402
from job_market_intelligence.retrieval.lexical import LexicalIndex  # noqa: E402
from job_market_intelligence.retrieval.search import HybridRetriever  # noqa: E402

RETRIEVAL_ROOT = Path(os.environ.get("RETRIEVAL_ROOT", ROOT / "data/private/retrieval"))
DOCUMENT_LABELS = Path(
    os.environ.get(
        "RETRIEVAL_DOCUMENT_LABELS",
        ROOT / "data/private/annotations_expanded/processed_adjudicated/document_labels.csv",
    )
)
# Defaults to the routed mode because that is what the ablation measured as
# best overall, at a twentieth of the latency of hybrid + reranking. Set
# RETRIEVAL_MODE=hybrid where traffic is dominated by queries that blend a
# lexical and a semantic constraint in one sentence -- routing is measurably
# weakest there, and the evidence for it is thin enough that the knob is worth
# exposing rather than baking the choice in. See README, Ablation results.
CONFIG = RetrievalConfig(
    chunk_tokens=int(os.environ.get("RETRIEVAL_CHUNK_TOKENS", "512")),
    mode=os.environ.get("RETRIEVAL_MODE", "routed"),  # type: ignore[arg-type]
    rerank=os.environ.get("RETRIEVAL_RERANK", "").lower() in {"1", "true", "yes"},
)

mcp = FastMCP("job-market-intelligence")
_retriever: HybridRetriever | None = None
_stats = JobStatsService(
    RETRIEVAL_ROOT / "job_market.duckdb",
    document_labels_path=DOCUMENT_LABELS if DOCUMENT_LABELS.exists() else None,
)


def get_retriever() -> HybridRetriever:
    """Load only the indexes the active configuration actually uses.

    Loading both unconditionally would make RETRIEVAL_MODE=lexical fail on a
    machine that has never built an embedding index, for a branch it was never
    going to run.
    """

    global _retriever
    if _retriever is None:
        label = CONFIG.index_label()
        chunks = pd.read_parquet(RETRIEVAL_ROOT / "chunks" / f"{label}.parquet")
        model_slug = CONFIG.embedding_model.rsplit("/", 1)[-1]
        _retriever = HybridRetriever(
            chunks,
            config=CONFIG,
            embeddings=(
                np.load(RETRIEVAL_ROOT / "index" / f"{label}.{model_slug}.npy")
                if CONFIG.uses_dense
                else None
            ),
            lexical_index=(
                LexicalIndex(RETRIEVAL_ROOT / "index" / f"{label}.lexical.duckdb")
                if CONFIG.uses_lexical
                else None
            ),
            encoder=(
                ChunkEncoder(CONFIG.embedding_model, cache_dir=RETRIEVAL_ROOT / "embedding_cache")
                if CONFIG.uses_dense
                else None
            ),
        )
    return _retriever


@mcp.tool()
def search_job_ads(
    query: str,
    top_k: int = 5,
    city: str = "",
    role: str = "",
    section: str = "",
) -> str:
    """Find passages from Australian job advertisements that describe something.

    USE THIS when the question is about *how advertisements are worded* or
    *which advertisements mention something*, and the answer is a quotation or
    a description. Examples: "what do these ads say about on-call duties?",
    "which roles mention dbt?", "how is stakeholder communication described?".

    DO NOT USE THIS for counts, shares, totals, averages, rankings or
    comparisons across the corpus -- "how many", "what proportion", "which is
    most common", "are there more X than Y". This tool returns only the
    handful of passages that match best; it has no view of the other ~700
    advertisements, so any number derived from its output will be wrong. Call
    `query_job_stats` for those instead.

    Returns ranked passages with the advertisement id, job title, company,
    city and the section heading the passage came from. Cite the advertisement
    id when you use a passage. The corpus mixes full-text advertisements from
    company career pages with ~500-character excerpts from an aggregator API;
    an excerpt ending mid-sentence is truncated at the source, and absence of a
    skill in an excerpt is not evidence the job does not require it.

    Args:
        query: What to look for, in natural language or as a tool name.
        top_k: Number of passages to return (1-20).
        city: Optional exact-match filter, e.g. "Sydney" or "Melbourne".
        role: Optional role filter, e.g. "Data Engineer", "Data Analyst".
        section: Optional section filter -- one of requirements,
            responsibilities, benefits, about, role, preferred, process, legal,
            intro, other. Use "requirements" to search only what a role asks for.
    """

    top_k = max(1, min(int(top_k), 20))
    filters = {
        key: value
        for key, value in (("city", city), ("analysis_role", role), ("section", section))
        if value
    }
    try:
        hits = get_retriever().search(query, top_k=top_k, filters=filters)
    except FileNotFoundError as exc:
        return json.dumps({"error": f"index not built: {exc}"})
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    warning = None
    if classify_query(query) == "aggregation":
        warning = (
            "This looks like a counting question. The passages below are a "
            "ranked sample, not a population. Do not state a count from them; "
            "call query_job_stats instead."
        )
    return json.dumps(
        {
            "query": query,
            "query_class": classify_query(query),
            "routing_note": warning or routing_rationale(query),
            "passages": [hit.to_dict() for hit in hits],
        },
        indent=2,
    )


@mcp.tool()
def query_job_stats(
    dimension: str,
    filter_field: str = "",
    filter_value: str = "",
    limit: int = 25,
) -> str:
    """Count advertisements across the whole corpus, grouped by one dimension.

    USE THIS whenever the answer is a number, a share, a ranking or a
    comparison: "how many ads mention Python?", "what proportion are in
    Sydney?", "which skill is most common?", "are there more engineer or
    analyst roles?". This runs SQL over every advertisement, so its counts are
    exact for the corpus.

    PREFER THIS OVER `search_job_ads` for any question containing how many,
    how much, what share, what proportion, what percentage, total, count,
    average, most/least common, rank, or a comparison between two groups.

    Args:
        dimension: What to group by. One of:
            role, city, seniority, source, company -- advertisement metadata;
            skill, skill_category, skill_context -- dictionary skill mentions;
            role_skill -- skill counts broken down by role;
            entry_fit, ai_signal, coding_signal, semantic_role -- hand-
            adjudicated labels, available only for the 244-advertisement
            labelled subset rather than the whole corpus.
        filter_field: Optional field to restrict to before grouping, e.g.
            "skill" with dimension "skill", or "city" with dimension "role".
        filter_value: The value for filter_field, e.g. "SQL" or "Sydney".
        limit: Maximum groups to return (1-200).

    Returns counts and each group's share of the relevant total. Skill counts
    measure dictionary matches in advertisement text, so they are mentions
    rather than verified requirements; label dimensions cover only the labelled
    subset. Both caveats are returned in the "note" field -- repeat them when
    the number matters.
    """

    try:
        result = _stats.query(
            dimension,
            filter_field=filter_field or None,
            filter_value=filter_value or None,
            limit=max(1, min(int(limit), 200)),
        )
    except ValueError as exc:
        return json.dumps(
            {"error": str(exc), "available_dimensions": sorted(DIMENSIONS)}, indent=2
        )
    return json.dumps(result.to_dict(), indent=2, default=str)


if __name__ == "__main__":
    mcp.run(transport="stdio")
