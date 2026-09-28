"""Capture fresh scores from raw evaluation inputs without reusing old indexes."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .chunking import build_chunk_frame
from .config import RetrievalConfig
from .embedding import MODEL_REVISIONS, ChunkEncoder
from .evaluation import dedupe_to_documents, score_query
from .golden import condense, judged_documents, load_queries, resolve_relevance
from .lexical import LexicalIndex
from .rerank import RERANK_REVISIONS, CrossEncoderReranker
from .search import HybridRetriever
from .validation import digest, json_digest, validate_scores, write_json


class DependencyUnavailable(RuntimeError):
    """A required local model, package or DuckDB extension is unavailable."""


def checked_config(payload: dict) -> RetrievalConfig:
    config = RetrievalConfig(**payload)
    if config.top_k < 10 or config.rrf_k <= 0 or not 0 <= config.overlap_ratio < 1:
        raise ValueError("configuration must support ndcg@10 and valid fusion/chunking")
    if config.uses_dense and config.embedding_model not in MODEL_REVISIONS:
        raise ValueError("embedding model must have a pinned revision")
    if config.rerank and config.rerank_model not in RERANK_REVISIONS:
        raise ValueError("reranker must have a pinned revision")
    return config


def capture(request: dict, run: Path) -> None:
    """Called in a subprocess so command, exit code and logs belong to this run."""
    config = checked_config(request["config"])
    inputs = {name: Path(path) for name, path in request["input_paths"].items()}
    if {name: digest(path) for name, path in inputs.items()} != request["input_hashes"]:
        raise ValueError("inputs changed before benchmark execution")
    if config.uses_lexical:
        try:
            with duckdb.connect() as connection:
                connection.execute("LOAD fts")
        except duckdb.Error as exc:
            raise DependencyUnavailable("DuckDB FTS must be installed before validation") from exc
    encoder = ChunkEncoder(config.embedding_model, device="cpu") if config.uses_dense else None
    reranker = CrossEncoderReranker(config.rerank_model, device="cpu") if config.rerank else None
    try:
        for model in (encoder, reranker):
            if model is not None:
                _ = model.model  # Offline environment is set by the runner.
    except (ImportError, OSError) as exc:
        raise DependencyUnavailable("required model or package is unavailable locally") from exc

    jobs = pd.read_csv(inputs["jobs"], dtype=str, keep_default_na=False)
    labels = pd.read_csv(inputs["document_labels"], dtype=str, keep_default_na=False)
    if jobs.empty or jobs.job_id.duplicated().any() or jobs.job_id.eq("").any():
        raise ValueError("corpus must contain unique nonempty job ids")
    with duckdb.connect(str(inputs["database"]), read_only=True) as connection:
        mentions = connection.execute("SELECT job_id, skill, category, context FROM skill_mentions")
        skill_mentions = mentions.fetchdf()
    skill_mentions["job_id"] = skill_mentions["job_id"].astype(str)
    golden = load_queries(inputs["queries"])
    judgements = []
    for query in golden:
        relevant = resolve_relevance(
            query, jobs=jobs, skill_mentions=skill_mentions, document_labels=labels
        )
        pool = judged_documents(query, jobs=jobs, document_labels=labels)
        if pool is not None:
            relevant &= pool
        if not relevant:
            raise ValueError(f"query has no relevant documents: {query.query_id}")
        judgements.append(
            {
                "query_id": query.query_id,
                "relevant": sorted(relevant),
                "judged": None if pool is None else sorted(pool),
            }
        )
    write_json(run / "judgements.json", judgements)
    chunks = build_chunk_frame(
        jobs, target_tokens=config.chunk_tokens, overlap_ratio=config.overlap_ratio
    )
    if chunks.empty or chunks.chunk_id.duplicated().any():
        raise ValueError("chunking must produce unique nonempty chunks")
    chunks.to_parquet(run / "chunks.parquet", index=False)
    lexical = LexicalIndex.build(chunks, run / "lexical.duckdb") if config.uses_lexical else None
    embeddings = None if encoder is None else encoder.encode(chunks.text.tolist())
    if embeddings is not None:
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must be finite")
        np.save(run / "embeddings.npy", embeddings)
    retriever = HybridRetriever(
        chunks,
        config=config,
        embeddings=embeddings,
        lexical_index=lexical,
        encoder=encoder,
        reranker=reranker,
    )
    chunk_to_job = dict(zip(chunks.chunk_id, chunks.job_id, strict=True))
    rows = []
    for query, judgement in zip(golden, judgements, strict=True):
        hits = retriever.search(query.query, top_k=config.retrieval_depth)
        if any(hit.chunk_id not in chunk_to_job for hit in hits):
            raise ValueError("retriever returned a chunk outside this run's corpus")
        ranked = dedupe_to_documents([hit.chunk_id for hit in hits], chunk_to_job)
        pool = None if judgement["judged"] is None else set(judgement["judged"])
        ranked = condense(ranked, pool)
        rows.append(
            {
                "query_id": query.query_id,
                "query_class": query.query_class,
                **score_query(set(judgement["relevant"]), ranked[: config.top_k]),
            }
        )
    scores = pd.DataFrame(rows)
    validate_scores(scores, request["expected_queries"], "ndcg@10")
    scores.to_csv(run / "scores.csv", index=False)
    write_json(
        run / "capture.json",
        {
            "run_id": request["run_id"],
            "config": asdict(config),
            "config_sha256": json_digest(asdict(config)),
            "model_revisions": {
                "embedding": MODEL_REVISIONS.get(config.embedding_model)
                if config.uses_dense
                else None,
                "reranker": RERANK_REVISIONS.get(config.rerank_model) if config.rerank else None,
            },
            "device": "cpu",
            "queries": len(scores),
            "documents": len(jobs),
            "judgements_sha256": digest(run / "judgements.json"),
            "scores_sha256": digest(run / "scores.csv"),
        },
    )


def worker(run: Path) -> int:
    if any(
        (run / name).exists()
        for name in (
            "complete.json",
            "judgements.json",
            "chunks.parquet",
            "scores.csv",
            "capture.json",
        )
    ):
        raise ValueError("benchmark worker requires a fresh run and cannot overwrite evidence")
    request = json.loads((run / "request.json").read_text())
    if request.get("run_id") != run.name:
        raise ValueError("benchmark request does not belong to this run")
    try:
        capture(request, run)
    except (DependencyUnavailable, ImportError) as exc:
        write_json(run / "worker_error.json", {"status": "NOT RUN", "reason": str(exc)})
        return 2
    return 0
