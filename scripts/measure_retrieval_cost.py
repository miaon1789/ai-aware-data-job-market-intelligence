#!/usr/bin/env python3
"""Measure where retrieval latency and answer cost actually go.

The ablation reports end-to-end latency per configuration. This breaks that
down by stage, so the reranker's cost can be attributed rather than inferred,
and estimates the token spend of answer synthesis at each retrieval depth.

Inference cost is the part of a retrieval system that grows quietly: the
reranker is free in dollars and expensive in milliseconds, while the answer
model is the reverse. Reporting them in the same table is the only way to make
that trade legible.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from job_market_intelligence.rag.prompts import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
    build_answer_prompt,
)
from job_market_intelligence.retrieval import dense  # noqa: E402
from job_market_intelligence.retrieval.chunking import estimate_tokens  # noqa: E402
from job_market_intelligence.retrieval.config import RetrievalConfig  # noqa: E402
from job_market_intelligence.retrieval.embedding import ChunkEncoder  # noqa: E402
from job_market_intelligence.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from job_market_intelligence.retrieval.golden import load_queries  # noqa: E402
from job_market_intelligence.retrieval.lexical import LexicalIndex  # noqa: E402
from job_market_intelligence.retrieval.rerank import CrossEncoderReranker  # noqa: E402
from job_market_intelligence.retrieval.search import HybridRetriever  # noqa: E402

REPORTS_ROOT = (ROOT / "reports").resolve()


def percentiles(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "mean_ms": round(float(statistics.fmean(values)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=ROOT / "eval/golden/queries.jsonl")
    parser.add_argument("--chunk-dir", type=Path, default=ROOT / "data/private/retrieval/chunks")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data/private/retrieval/index")
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "data/private/retrieval/embedding_cache"
    )
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports/retrieval")
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--answer-passages", type=int, default=5)
    parser.add_argument(
        "--input-cost-per-million",
        type=float,
        default=3.0,
        help="USD per million input tokens for the answer model",
    )
    parser.add_argument("--output-cost-per-million", type=float, default=15.0)
    parser.add_argument("--assumed-output-tokens", type=int, default=250)
    args = parser.parse_args()

    resolved_reports = args.reports_dir.resolve()
    if resolved_reports != REPORTS_ROOT and REPORTS_ROOT not in resolved_reports.parents:
        parser.error("cost reports must stay under reports/")

    config = RetrievalConfig(chunk_tokens=args.chunk_tokens, mode="hybrid")
    label = config.index_label()
    chunks = pd.read_parquet(args.chunk_dir / f"{label}.parquet")
    model_slug = config.embedding_model.rsplit("/", 1)[-1]
    embeddings = np.load(args.index_dir / f"{label}.{model_slug}.npy")
    lexical = LexicalIndex(args.index_dir / f"{label}.lexical.duckdb")
    queries = [query.query for query in load_queries(args.queries)]

    stages: dict[str, list[float]] = {
        "query_encode": [],
        "dense_scan": [],
        "lexical_bm25": [],
        "fusion": [],
        "rerank": [],
    }

    warm_encoder = ChunkEncoder(config.embedding_model, cache_dir=args.cache_dir)
    cold_encoder = ChunkEncoder(config.embedding_model)
    reranker = CrossEncoderReranker(config.rerank_model)
    # Pay each model load once so it is not attributed to the first query.
    cold_encoder.encode(["warm up"], is_query=True)
    warm_encoder.encode(["warm up"], is_query=True)
    reranker.rerank("warm up", [("c", "warm up text")], top_k=1)

    # Populate the cache first; a cache measured on its misses is measuring the
    # model, not the cache.
    for query in queries:
        warm_encoder.encode([query], is_query=True)

    cold_encode: list[float] = []
    warm_encode: list[float] = []
    for query in queries:
        started = time.perf_counter()
        vector = cold_encoder.encode([query], is_query=True)[0]
        cold_encode.append((time.perf_counter() - started) * 1000)
        stages["query_encode"].append(cold_encode[-1])

        started = time.perf_counter()
        warm_encoder.encode([query], is_query=True)
        warm_encode.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        dense_hits = dense.search(vector, embeddings, top_k=config.retrieval_depth)
        stages["dense_scan"].append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        lexical_hits = lexical.search(query, top_k=config.retrieval_depth)
        stages["lexical_bm25"].append((time.perf_counter() - started) * 1000)

        dense_ids = [str(chunks.at[row, "chunk_id"]) for row, _ in dense_hits]
        lexical_ids = [chunk_id for chunk_id, _ in lexical_hits]
        started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            {"dense": dense_ids, "lexical": lexical_ids}, k=config.rrf_k
        )
        stages["fusion"].append((time.perf_counter() - started) * 1000)

        row_of = {chunk_id: index for index, chunk_id in enumerate(chunks["chunk_id"])}
        shortlist = [
            (chunk_id, str(chunks.at[row_of[chunk_id], "text"]))
            for chunk_id, _ in fused[: config.candidate_pool]
            if chunk_id in row_of
        ]
        started = time.perf_counter()
        reranker.rerank(query, shortlist, top_k=config.top_k)
        stages["rerank"].append((time.perf_counter() - started) * 1000)

    warm_encoder.save_cache()

    # Answer-synthesis token cost, measured on the passages retrieval returns.
    retriever = HybridRetriever(
        chunks, config=config, embeddings=embeddings, lexical_index=lexical,
        encoder=warm_encoder,
    )
    prompt_tokens: list[int] = []
    for query in queries:
        passages = [hit.to_dict() for hit in retriever.search(query, top_k=args.answer_passages)]
        prompt = build_answer_prompt(query, passages)
        prompt_tokens.append(estimate_tokens(ANSWER_SYSTEM_PROMPT) + estimate_tokens(prompt))

    mean_input = statistics.fmean(prompt_tokens)
    cost_per_answer = (
        mean_input / 1_000_000 * args.input_cost_per_million
        + args.assumed_output_tokens / 1_000_000 * args.output_cost_per_million
    )

    stage_rows = [
        {"stage": stage, "queries": len(values), **percentiles(values)}
        for stage, values in stages.items()
        if values
    ]
    retrieval_no_rerank = sum(
        statistics.fmean(stages[stage])
        for stage in ("query_encode", "dense_scan", "lexical_bm25", "fusion")
    )
    report = {
        "configuration": config.to_dict(),
        "queries": len(queries),
        "corpus_chunks": int(len(chunks)),
        "dense_index_megabytes": round(embeddings.nbytes / 1_048_576, 2),
        "stages": stage_rows,
        "retrieval_mean_ms_without_rerank": round(retrieval_no_rerank, 2),
        "rerank_mean_ms": round(statistics.fmean(stages["rerank"]), 2),
        "rerank_share_of_total": round(
            statistics.fmean(stages["rerank"])
            / (retrieval_no_rerank + statistics.fmean(stages["rerank"])),
            3,
        ),
        "embedding_cache": {
            "note": (
                "Cold encodes the query through the transformer; warm is a repeat "
                "of the same query served from the on-disk vector cache."
            ),
            "cold_mean_ms": round(statistics.fmean(cold_encode), 2),
            "warm_mean_ms": round(statistics.fmean(warm_encode), 3),
            "speedup": round(
                statistics.fmean(cold_encode) / max(1e-6, statistics.fmean(warm_encode)), 1
            ),
        },
        "answer_synthesis": {
            "passages_per_answer": args.answer_passages,
            "mean_input_tokens": round(mean_input),
            "p95_input_tokens": int(np.percentile(prompt_tokens, 95)),
            "assumed_output_tokens": args.assumed_output_tokens,
            "usd_per_answer": round(cost_per_answer, 5),
            "usd_per_thousand_answers": round(cost_per_answer * 1000, 2),
            "pricing_assumption_usd_per_million": {
                "input": args.input_cost_per_million,
                "output": args.output_cost_per_million,
            },
            "note": (
                "Token counts are the four-characters-per-token approximation used "
                "throughout the chunker, not a tokeniser count. Reranking adds no "
                "token cost: the cross-encoder runs locally."
            ),
        },
    }

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stage_rows).to_csv(args.reports_dir / "cost_latency_stages.csv", index=False)
    (args.reports_dir / "cost_latency.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
