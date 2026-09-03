#!/usr/bin/env python3
"""Run the retrieval ablation grid and write the evaluation reports.

Every metric is reported three ways: overall, split by query class, and split
by whether the advertisement carries full text or a truncated excerpt. The
by-class split is the point of the exercise -- an overall mean hides a gain
that lands entirely in one class -- and the truncation split is what makes the
chunk-size axis interpretable on a corpus where two thirds of the documents
cannot be chunked at all.

Only aggregate numbers leave `data/private/`; no advertisement text or id is
written to `reports/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from job_market_intelligence.retrieval.config import (  # noqa: E402
    CHUNK_SIZES,
    EMBEDDING_MODELS,
    RETRIEVAL_MODES,
    RetrievalConfig,
)
from job_market_intelligence.retrieval.evaluation import (  # noqa: E402
    DEFAULT_KS,
    aggregate_scores,
    dedupe_to_documents,
    paired_comparison,
    resolution_limit,
    routing_break_even,
    score_query,
)
from job_market_intelligence.retrieval.golden import condense, load_queries  # noqa: E402
from job_market_intelligence.retrieval.lexical import LexicalIndex  # noqa: E402
from job_market_intelligence.retrieval.search import HybridRetriever  # noqa: E402

REPORTS_ROOT = (ROOT / "reports").resolve()


def model_slug(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1]


def load_resolved(path: Path) -> dict[str, dict[str, object]]:
    resolved: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            resolved[row["query_id"]] = row
    return resolved


def build_grid(args) -> list[RetrievalConfig]:
    """The ablation grid: one axis moves at a time from a fixed baseline.

    A full cross-product would be 3 x 3 x 2 x 2 = 36 configurations, most of
    which answer no question anyone asked. Each axis is instead varied against
    a common baseline, which is what makes a row-to-row difference readable as
    the effect of that one change.
    """

    baseline = RetrievalConfig(
        chunk_tokens=args.baseline_chunk_tokens,
        mode="hybrid",
        embedding_model=EMBEDDING_MODELS[0],
        rerank=False,
        candidate_pool=args.candidate_pool,
        top_k=args.top_k,
    )
    configs = {baseline.label(): baseline}
    for chunk_tokens in CHUNK_SIZES:
        candidate = baseline.with_(chunk_tokens=chunk_tokens)
        configs.setdefault(candidate.label(), candidate)
    for mode in RETRIEVAL_MODES:
        candidate = baseline.with_(mode=mode)
        configs.setdefault(candidate.label(), candidate)
    for model in EMBEDDING_MODELS:
        candidate = baseline.with_(embedding_model=model)
        configs.setdefault(candidate.label(), candidate)
    for mode, rerank in product(RETRIEVAL_MODES, (True,)):
        candidate = baseline.with_(mode=mode, rerank=rerank)
        configs.setdefault(candidate.label(), candidate)
    return list(configs.values())


def routing_accuracy(golden) -> dict[str, object]:
    """How often the heuristic router agrees with the golden query classes.

    On this set the answer is 100%, and that number means very little: the
    router's patterns and these queries were written by the same hand. The
    honest estimate is the held-out probe set, reported alongside.
    """

    from job_market_intelligence.rag.routing import classify_query

    agree = sum(classify_query(query.query) == query.query_class for query in golden)
    return {
        "queries": len(golden),
        "agreement_with_golden_classes": round(agree / max(1, len(golden)), 4),
        "caveat": (
            "The router was written against these queries, so this is an upper "
            "bound. eval/golden/routing_heldout.jsonl is the held-out estimate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=ROOT / "eval/golden/queries.jsonl")
    parser.add_argument(
        "--resolved", type=Path, default=ROOT / "data/private/retrieval/golden/resolved.jsonl"
    )
    parser.add_argument("--chunk-dir", type=Path, default=ROOT / "data/private/retrieval/chunks")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data/private/retrieval/index")
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "data/private/retrieval/embedding_cache"
    )
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports/retrieval")
    parser.add_argument(
        "--private-output", type=Path, default=ROOT / "data/private/retrieval/ablation"
    )
    parser.add_argument("--baseline-chunk-tokens", type=int, default=512)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()

    if REPORTS_ROOT not in args.reports_dir.resolve().parents and (
        args.reports_dir.resolve() != REPORTS_ROOT
    ):
        parser.error("aggregate reports must stay under reports/")

    golden = load_queries(args.queries)
    resolved = load_resolved(args.resolved)
    missing = [query.query_id for query in golden if query.query_id not in resolved]
    if missing:
        parser.error(f"unresolved queries (run build_golden_set.py): {', '.join(missing)}")

    configs = build_grid(args)
    chunk_frames: dict[int, pd.DataFrame] = {}
    per_query_rows: list[dict[str, object]] = []
    timings: list[dict[str, object]] = []

    for config in configs:
        chunk_label = config.index_label()
        if config.chunk_tokens not in chunk_frames:
            chunk_frames[config.chunk_tokens] = pd.read_parquet(
                args.chunk_dir / f"{chunk_label}.parquet"
            )
        chunks = chunk_frames[config.chunk_tokens]
        chunk_to_job = dict(zip(chunks["chunk_id"], chunks["job_id"].astype(str), strict=True))
        truncated_by_job = dict(
            zip(chunks["job_id"].astype(str), chunks["is_truncated"], strict=True)
        )

        embeddings = None
        if config.uses_dense:
            embeddings = np.load(
                args.index_dir / f"{chunk_label}.{model_slug(config.embedding_model)}.npy"
            )
        lexical_index = (
            LexicalIndex(args.index_dir / f"{chunk_label}.lexical.duckdb")
            if config.uses_lexical
            else None
        )
        retriever = HybridRetriever(
            chunks,
            config=config,
            embeddings=embeddings,
            lexical_index=lexical_index,
            encoder=None,
        )
        if config.uses_dense:
            from job_market_intelligence.retrieval.embedding import ChunkEncoder

            retriever._encoder = ChunkEncoder(config.embedding_model, cache_dir=args.cache_dir)

        latencies: list[float] = []
        for query in golden:
            judgement = resolved[query.query_id]
            relevant = set(judgement["relevant_job_ids"])
            pool = set(judgement["judged_job_ids"]) or None

            started = time.perf_counter()
            hits = retriever.search(query.query, top_k=config.retrieval_depth)
            latencies.append((time.perf_counter() - started) * 1000)

            ranked = dedupe_to_documents([hit.chunk_id for hit in hits], chunk_to_job)
            # Unjudged advertisements are removed rather than counted as misses.
            ranked = condense(ranked, pool)
            scores = score_query(relevant, ranked[: config.top_k], ks=DEFAULT_KS)

            share_full_text = (
                float(
                    np.mean([not truncated_by_job.get(job_id, True) for job_id in relevant])
                )
                if relevant
                else float("nan")
            )
            per_query_rows.append(
                {
                    "config": config.label(),
                    "chunk_tokens": config.chunk_tokens,
                    "mode": config.mode,
                    "embedding_model": model_slug(config.embedding_model)
                    if config.uses_dense
                    else "none",
                    "rerank": config.rerank,
                    "query_id": query.query_id,
                    "query_class": query.query_class,
                    "outside_skill_dictionary": query.outside_skill_dictionary,
                    "relevant_documents": len(relevant),
                    "relevant_share_full_text": share_full_text,
                    "max_recall_at_10": min(config.top_k, len(relevant)) / max(1, len(relevant)),
                    **scores,
                }
            )

        timings.append(
            {
                "config": config.label(),
                **config.to_dict(),
                "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
                "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
                "latency_mean_ms": round(float(np.mean(latencies)), 2),
            }
        )
        print(f"  ran {config.label()}", file=sys.stderr)

    per_query = pd.DataFrame(per_query_rows)
    args.private_output.mkdir(parents=True, exist_ok=True)
    args.private_output.chmod(0o700)
    per_query.to_csv(args.private_output / "per_query_scores.csv", index=False)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    bootstrap = {"iterations": args.bootstrap_iterations}

    overall = pd.concat(
        [
            aggregate_scores(part, **bootstrap).assign(config=label)
            for label, part in per_query.groupby("config")
        ],
        ignore_index=True,
    )
    overall = overall[["config", *[c for c in overall.columns if c != "config"]]]
    overall.to_csv(args.reports_dir / "ablation_overall.csv", index=False)

    by_class = pd.concat(
        [
            aggregate_scores(part, group_columns=["query_class"], **bootstrap).assign(
                config=label
            )
            for label, part in per_query.groupby("config")
        ],
        ignore_index=True,
    )
    # recall@k is bounded above by min(k, |relevant|) / |relevant|. Publishing
    # the ceiling next to the score is what turns a low aggregation number from
    # an apparent failure into the structural fact it is.
    ceilings = (
        per_query.groupby(["config", "query_class"])
        .agg(
            recall_at_10_ceiling=("max_recall_at_10", "mean"),
            mean_relevant_documents=("relevant_documents", "mean"),
        )
        .round(4)
        .reset_index()
    )
    by_class = by_class.merge(ceilings, on=["config", "query_class"], how="left")
    by_class = by_class[["config", "query_class", *[
        c for c in by_class.columns if c not in {"config", "query_class"}
    ]]]
    by_class.to_csv(args.reports_dir / "ablation_by_query_class.csv", index=False)

    lexical_only = per_query[per_query["query_class"].eq("lexical")]
    by_dictionary = pd.concat(
        [
            aggregate_scores(
                part, group_columns=["outside_skill_dictionary"], **bootstrap
            ).assign(config=label)
            for label, part in lexical_only.groupby("config")
        ],
        ignore_index=True,
    )
    by_dictionary.to_csv(args.reports_dir / "ablation_lexical_dictionary_bias.csv", index=False)

    pd.DataFrame(timings).to_csv(args.reports_dir / "ablation_latency.csv", index=False)

    # Head-to-head tests. Separate per-configuration intervals overlap for
    # systems that a paired test tells apart, so the decisions that matter are
    # tested directly rather than read off the summary table.
    bm25 = RetrievalConfig(chunk_tokens=args.baseline_chunk_tokens, mode="lexical").label()
    dense_only = RetrievalConfig(chunk_tokens=args.baseline_chunk_tokens, mode="dense").label()
    hybrid = RetrievalConfig(chunk_tokens=args.baseline_chunk_tokens, mode="hybrid").label()
    hybrid_rerank = RetrievalConfig(
        chunk_tokens=args.baseline_chunk_tokens, mode="hybrid", rerank=True
    ).label()
    routed = RetrievalConfig(chunk_tokens=args.baseline_chunk_tokens, mode="routed").label()

    comparisons = []
    for left, right in (
        (bm25, hybrid),
        (bm25, hybrid_rerank),
        (routed, hybrid),
        (routed, hybrid_rerank),
        (dense_only, hybrid),
    ):
        for groups in ((), ("query_class",)):
            comparisons.append(
                paired_comparison(per_query, left, right, group_columns=groups)
            )
    pd.concat(comparisons, ignore_index=True).to_csv(
        args.reports_dir / "paired_comparisons.csv", index=False
    )

    break_even = routing_break_even(
        per_query,
        branch_for_class={
            "lexical": bm25,
            "semantic": dense_only,
            # For a mixed query the *correct* answer is fusion -- both halves of
            # the query need a branch. The router cannot emit "mixed", so these
            # are structurally misroutable and drag the routed configuration
            # down exactly as they should.
            "mixed": hybrid,
            "aggregation": hybrid,
        },
        fallback_config=hybrid,
        fusion_config=hybrid,
    )

    baseline_label = RetrievalConfig(
        chunk_tokens=args.baseline_chunk_tokens,
        mode="hybrid",
        top_k=args.top_k,
        candidate_pool=args.candidate_pool,
    ).label()
    baseline_rows = per_query[per_query["config"].eq(baseline_label)]
    summary = {
        "configurations": len(configs),
        "queries": len(golden),
        "corpus_chunks": {
            str(tokens): int(len(frame)) for tokens, frame in chunk_frames.items()
        },
        "baseline_config": baseline_label,
        "resolution_limit_recall_at_10": resolution_limit(
            baseline_rows, "recall@10", iterations=args.bootstrap_iterations
        ),
        "heuristic_routing": routing_accuracy(golden),
        "routing_break_even": break_even,
        "aggregation_recall_ceiling": round(
            float(
                per_query[per_query["query_class"].eq("aggregation")]["max_recall_at_10"].mean()
            ),
            4,
        ),
        "reports": sorted(
            str(path.relative_to(ROOT)) for path in args.reports_dir.glob("ablation_*.csv")
        ),
    }
    (args.reports_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
