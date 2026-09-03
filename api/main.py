#!/usr/bin/env python3
"""Local FastAPI service over the retrieval corpus.

Bound to the loopback interface by default and never deployed publicly: the
index it serves is built from licensed advertisement text that the project's
data terms keep private. The public artefacts of this work are the evaluation
tables, not the corpus.

Endpoints:
    POST /search        ranked passages with scores
    POST /answer        routed answer -- retrieval or SQL, with the reason
    POST /stats         aggregate counts over an allowlisted dimension
    POST /reindex       asynchronous rebuild; returns a job id immediately
    GET  /jobs/{id}     status of a background job
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_market_intelligence.rag.answer import synthesise_answer  # noqa: E402
from job_market_intelligence.rag.routing import (  # noqa: E402
    classify_query,
    route,
    routing_rationale,
)
from job_market_intelligence.rag.stats import JobStatsService  # noqa: E402
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
DEFAULT_CONFIG = RetrievalConfig(
    chunk_tokens=int(os.environ.get("RETRIEVAL_CHUNK_TOKENS", "512")),
    mode=os.environ.get("RETRIEVAL_MODE", "routed"),  # type: ignore[arg-type]
    rerank=os.environ.get("RETRIEVAL_RERANK", "").lower() in {"1", "true", "yes"},
)


@dataclass
class BackgroundJob:
    job_id: str
    kind: str
    status: Literal["queued", "running", "succeeded", "failed"] = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "detail": self.detail,
        }


class JobRegistry:
    """In-memory job table guarded by a lock; one process, one registry."""

    def __init__(self, max_workers: int = 1) -> None:
        self._jobs: dict[str, BackgroundJob] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, kind: str, function, *args) -> BackgroundJob:
        job = BackgroundJob(job_id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._jobs[job.job_id] = job
        self._pool.submit(self._run, job, function, *args)
        return job

    def _run(self, job: BackgroundJob, function, *args) -> None:
        with self._lock:
            job.status = "running"
        try:
            detail = function(*args)
            with self._lock:
                job.status = "succeeded"
                job.detail = str(detail)[:2000]
        except Exception as exc:  # noqa: BLE001 - surfaced through the job record
            with self._lock:
                job.status = "failed"
                job.detail = f"{type(exc).__name__}: {exc}"[:2000]
        finally:
            with self._lock:
                job.finished_at = datetime.now(UTC).isoformat()

    def get(self, job_id: str) -> BackgroundJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[BackgroundJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class RetrievalState:
    """Holds the loaded index; swapped atomically after a rebuild."""

    def __init__(self, config: RetrievalConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._retriever: HybridRetriever | None = None
        self._lock = threading.Lock()

    def load(self, config: RetrievalConfig | None = None) -> HybridRetriever:
        config = config or self.config
        label = config.index_label()
        chunk_path = RETRIEVAL_ROOT / "chunks" / f"{label}.parquet"
        if not chunk_path.exists():
            raise FileNotFoundError(
                f"missing chunk file {chunk_path}; run scripts/build_chunks.py first"
            )
        chunks = pd.read_parquet(chunk_path)
        model_slug = config.embedding_model.rsplit("/", 1)[-1]
        embeddings = (
            np.load(RETRIEVAL_ROOT / "index" / f"{label}.{model_slug}.npy")
            if config.uses_dense
            else None
        )
        lexical = (
            LexicalIndex(RETRIEVAL_ROOT / "index" / f"{label}.lexical.duckdb")
            if config.uses_lexical
            else None
        )
        return HybridRetriever(
            chunks,
            config=config,
            embeddings=embeddings,
            lexical_index=lexical,
            encoder=ChunkEncoder(
                config.embedding_model,
                cache_dir=RETRIEVAL_ROOT / "embedding_cache",
            )
            if config.uses_dense
            else None,
        )

    @property
    def retriever(self) -> HybridRetriever:
        with self._lock:
            if self._retriever is None:
                self._retriever = self.load()
            return self._retriever

    def reload(self, config: RetrievalConfig | None = None) -> str:
        """Swap in a freshly built index, optionally under a new configuration.

        The build happens before the lock is taken, so a failure leaves the
        running index and its configuration untouched. Reloading without
        updating the configuration was a real bug: a request to rebuild at 256
        tokens rebuilt the chunks and then reloaded the 512-token index, and
        the service reported success while serving the old one.
        """

        target = config or self.config
        fresh = self.load(target)
        with self._lock:
            self.config = target
            self._retriever = fresh
        return f"reloaded {len(fresh.chunks)} chunks for {target.index_label()}"


state = RetrievalState()
registry = JobRegistry()
stats_service = JobStatsService(
    RETRIEVAL_ROOT / "job_market.duckdb",
    document_labels_path=DOCUMENT_LABELS if DOCUMENT_LABELS.exists() else None,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    registry.shutdown()


app = FastAPI(
    title="Job-ad retrieval service",
    description=__doc__,
    version="0.1.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    city: str = ""
    role: str = ""
    section: str = ""


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    city: str = ""
    role: str = ""


class StatsRequest(BaseModel):
    dimension: str
    filter_field: str = ""
    filter_value: str = ""
    limit: int = Field(default=25, ge=1, le=200)


class ReindexRequest(BaseModel):
    chunk_tokens: int = Field(default=512, ge=32, le=4096)
    rebuild_chunks: bool = True


def _filters(city: str = "", role: str = "", section: str = "") -> dict[str, str]:
    return {
        key: value
        for key, value in (("city", city), ("analysis_role", role), ("section", section))
        if value
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "config": state.config.to_dict(),
        "retrieval_root": str(RETRIEVAL_ROOT),
        "stats_dimensions": stats_service.available_dimensions(),
    }


@app.post("/search")
def search(request: SearchRequest) -> dict[str, object]:
    try:
        hits = state.retriever.search(
            request.query,
            top_k=request.top_k,
            filters=_filters(request.city, request.role, request.section),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {
        "query": request.query,
        "query_class": classify_query(request.query),
        "results": [hit.to_dict() for hit in hits],
    }


@app.post("/stats")
def stats(request: StatsRequest) -> dict[str, object]:
    try:
        result = stats_service.query(
            request.dimension,
            filter_field=request.filter_field or None,
            filter_value=request.filter_value or None,
            limit=request.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return result.to_dict()


@app.post("/answer")
def answer(request: AnswerRequest) -> dict[str, object]:
    """Route the question, then answer it with the tool that can be correct."""

    tool = route(request.question)
    payload: dict[str, object] = {
        "question": request.question,
        "query_class": classify_query(request.question),
        "routed_to": tool,
        "rationale": routing_rationale(request.question),
    }

    if tool == "query_job_stats":
        payload["answer"] = (
            "This is a population question. Retrieval returns a handful of "
            "passages and cannot count, so it is answered from the analysis "
            "database instead. Call /stats with the dimension you need."
        )
        payload["available_dimensions"] = stats_service.available_dimensions()
        return payload

    try:
        hits = state.retriever.search(
            request.question, top_k=request.top_k, filters=_filters(request.city, request.role)
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None

    passages = [hit.to_dict() for hit in hits]
    grounded = synthesise_answer(request.question, passages)
    payload["passages"] = passages
    payload["grounded_answer"] = grounded.to_dict()
    return payload


def _reindex(chunk_tokens: int, rebuild_chunks: bool) -> str:
    """Rebuild artefacts in a subprocess, then hot-swap the loaded index."""

    steps: list[str] = []
    if rebuild_chunks:
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(ROOT / "scripts/build_chunks.py"),
                "--chunk-tokens",
                str(chunk_tokens),
            ],
            check=True,
            capture_output=True,
        )
        steps.append("chunks")
    subprocess.run(  # noqa: S603
        [sys.executable, str(ROOT / "scripts/build_retrieval_index.py")],
        check=True,
        capture_output=True,
    )
    steps.append("index")
    # Reload under the configuration that was just built, not the one that was
    # running when the request arrived.
    steps.append(state.reload(state.config.with_(chunk_tokens=chunk_tokens)))
    return "; ".join(steps)


@app.post("/reindex", status_code=202)
def reindex(request: ReindexRequest) -> dict[str, object]:
    """Start a rebuild and return immediately with a job id to poll."""

    job = registry.submit("reindex", _reindex, request.chunk_tokens, request.rebuild_chunks)
    return {"job": job.to_dict(), "poll": f"/jobs/{job.job_id}"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, object]:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    return job.to_dict()


@app.get("/jobs")
def list_jobs() -> dict[str, object]:
    return {"jobs": [job.to_dict() for job in registry.list()]}


if __name__ == "__main__":
    import uvicorn

    # Loopback only: the index contains licensed advertisement text.
    uvicorn.run(app, host="127.0.0.1", port=8000)
