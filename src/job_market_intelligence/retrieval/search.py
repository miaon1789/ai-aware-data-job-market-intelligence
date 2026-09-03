"""Assembles chunking, dense, lexical, fusion and reranking into one searcher."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..rag.routing import classify_query
from . import dense
from .config import RetrievalConfig
from .embedding import ChunkEncoder
from .fusion import reciprocal_rank_fusion
from .lexical import FILTERABLE_COLUMNS, LexicalIndex
from .rerank import CrossEncoderReranker


@dataclass(frozen=True)
class RetrievedChunk:
    """One result, carrying enough context to be cited without a second lookup."""

    chunk_id: str
    job_id: str
    rank: int
    score: float
    section: str
    heading: str
    body: str
    title: str
    company: str
    city: str
    source: str
    analysis_role: str
    posted_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "job_id": self.job_id,
            "rank": self.rank,
            "score": round(self.score, 6),
            "section": self.section,
            "heading": self.heading,
            "body": self.body,
            "title": self.title,
            "company": self.company,
            "city": self.city,
            "source": self.source,
            "analysis_role": self.analysis_role,
            "posted_at": self.posted_at,
        }


class HybridRetriever:
    """Dense, lexical or RRF-fused retrieval with optional cross-encoder rerank."""

    def __init__(
        self,
        chunks: pd.DataFrame,
        *,
        config: RetrievalConfig,
        embeddings: np.ndarray | None = None,
        lexical_index: LexicalIndex | None = None,
        encoder: ChunkEncoder | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        if "chunk_id" not in chunks.columns:
            raise ValueError("chunks must contain a chunk_id column")
        if config.uses_dense and embeddings is None:
            raise ValueError("dense retrieval requires an embedding matrix")
        if config.uses_lexical and lexical_index is None:
            raise ValueError("lexical retrieval requires a lexical index")
        if config.uses_dense and embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("embedding matrix and chunk frame must align row for row")

        self.chunks = chunks.reset_index(drop=True)
        self.config = config
        self.embeddings = embeddings
        self.lexical_index = lexical_index
        self._encoder = encoder
        self._reranker = reranker
        self._row_of = {
            chunk_id: index for index, chunk_id in enumerate(self.chunks["chunk_id"])
        }

    @classmethod
    def from_artefacts(
        cls,
        chunks: pd.DataFrame,
        *,
        config: RetrievalConfig,
        embedding_path: str | Path | None = None,
        lexical_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> HybridRetriever:
        embeddings = np.load(embedding_path) if embedding_path is not None else None
        index = LexicalIndex(lexical_path) if lexical_path is not None else None
        encoder = (
            ChunkEncoder(config.embedding_model, cache_dir=cache_dir)
            if config.uses_dense
            else None
        )
        return cls(
            chunks,
            config=config,
            embeddings=embeddings,
            lexical_index=index,
            encoder=encoder,
        )

    @property
    def encoder(self) -> ChunkEncoder:
        if self._encoder is None:
            self._encoder = ChunkEncoder(self.config.embedding_model)
        return self._encoder

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.config.rerank_model)
        return self._reranker

    def _allowed_rows(self, filters: Mapping[str, str] | None) -> np.ndarray | None:
        if not filters:
            return None
        mask = pd.Series(True, index=self.chunks.index)
        for column, value in filters.items():
            if column not in FILTERABLE_COLUMNS:
                raise ValueError(f"cannot filter on {column!r}")
            if value:
                mask &= self.chunks[column].astype(str).eq(str(value))
        return np.flatnonzero(mask.to_numpy())

    def _dense_ranking(
        self, query: str, *, pool: int, filters: Mapping[str, str] | None
    ) -> list[str]:
        vector = self.encoder.encode([query], is_query=True)[0]
        hits = dense.search(
            vector, self.embeddings, top_k=pool, allowed_rows=self._allowed_rows(filters)
        )
        return [str(self.chunks.at[row, "chunk_id"]) for row, _ in hits]

    def _lexical_ranking(
        self, query: str, *, pool: int, filters: Mapping[str, str] | None
    ) -> list[str]:
        hits = self.lexical_index.search(query, top_k=pool, filters=filters)
        return [chunk_id for chunk_id, _ in hits if chunk_id in self._row_of]

    # Which branch answers which query shape. Fusing is what you do when you
    # cannot tell the query apart; the ablation shows each branch beats the
    # fusion on the class it owns, so where the class is knowable, route.
    ROUTED_BRANCHES = {
        "lexical": ("lexical",),
        "semantic": ("dense",),
        # Counting questions do not reach retrieval in production -- they go to
        # query_job_stats. Fusion is the conservative fallback if one does.
        "aggregation": ("dense", "lexical"),
    }

    def _active_branches(self, query: str) -> tuple[str, ...]:
        """Return the retrieval branches to run for this query."""

        if self.config.mode == "routed":
            return self.ROUTED_BRANCHES[classify_query(query)]
        branches = []
        if self.config.uses_dense:
            branches.append("dense")
        if self.config.uses_lexical:
            branches.append("lexical")
        return tuple(branches)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: Mapping[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the best chunks for ``query`` under the active config."""

        limit = top_k or self.config.top_k
        depth = max(self.config.retrieval_depth, limit)

        branches = self._active_branches(query)
        rankings: dict[str, Sequence[str]] = {}
        if "dense" in branches:
            rankings["dense"] = self._dense_ranking(query, pool=depth, filters=filters)
        if "lexical" in branches:
            rankings["lexical"] = self._lexical_ranking(query, pool=depth, filters=filters)

        if len(rankings) == 1:
            ordered = [(chunk_id, 1.0 / (rank + 1)) for rank, chunk_id in enumerate(
                next(iter(rankings.values()))
            )]
        else:
            ordered = reciprocal_rank_fusion(rankings, k=self.config.rrf_k)

        if self.config.rerank and ordered:
            head = ordered[: self.config.candidate_pool]
            tail = ordered[self.config.candidate_pool :]
            shortlist = [
                (chunk_id, str(self.chunks.at[self._row_of[chunk_id], "text"]))
                for chunk_id, _ in head
                if chunk_id in self._row_of
            ]
            reranked = self.reranker.rerank(query, shortlist, top_k=len(shortlist))
            # The cross-encoder reorders the head it was given; everything below
            # keeps its fused order so that deep recall is unaffected.
            ordered = reranked + tail

        results: list[RetrievedChunk] = []
        for rank, (chunk_id, score) in enumerate(ordered[:limit], start=1):
            row = self.chunks.iloc[self._row_of[chunk_id]]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    job_id=str(row["job_id"]),
                    rank=rank,
                    score=float(score),
                    section=str(row.get("section", "")),
                    heading=str(row.get("heading", "")),
                    body=str(row.get("body", "")),
                    title=str(row.get("title", "")),
                    company=str(row.get("company", "")),
                    city=str(row.get("city", "")),
                    source=str(row.get("source", "")),
                    analysis_role=str(row.get("analysis_role", "")),
                    posted_at=str(row.get("posted_at", "")),
                )
            )
        return results
