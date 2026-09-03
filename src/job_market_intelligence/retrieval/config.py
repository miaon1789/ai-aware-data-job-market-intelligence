"""The ablation surface, expressed as one immutable configuration object."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .embedding import ALTERNATE_MODEL, DEFAULT_MODEL
from .rerank import DEFAULT_RERANK_MODEL

RetrievalMode = Literal["dense", "lexical", "hybrid", "routed"]

CHUNK_SIZES = (256, 512, 1024)
RETRIEVAL_MODES: tuple[RetrievalMode, ...] = ("dense", "lexical", "hybrid", "routed")
EMBEDDING_MODELS = (DEFAULT_MODEL, ALTERNATE_MODEL)


@dataclass(frozen=True)
class RetrievalConfig:
    """One point in the ablation grid.

    Two settings are deliberately *not* ablated and are documented as fixed
    choices: chunks are always indexed with their advertisement title and
    section heading prepended, and RRF always uses k=60. Both were fixed before
    the golden set was built so that neither could be tuned against it.
    """

    chunk_tokens: int = 512
    overlap_ratio: float = 0.1
    mode: RetrievalMode = "hybrid"
    embedding_model: str = DEFAULT_MODEL
    rrf_k: int = 60
    rerank: bool = False
    rerank_model: str = DEFAULT_RERANK_MODEL
    # Chunks pulled from each branch before fusion. Deep enough that collapsing
    # chunks to advertisements still leaves more than top_k distinct ads at
    # every chunk size -- otherwise a smaller window would score worse purely
    # because its chunks crowd each other out of a fixed-depth pool, and the
    # chunk-size axis would be measuring the pool, not the chunking.
    retrieval_depth: int = 200
    # Head of the fused list handed to the cross-encoder.
    candidate_pool: int = 50
    top_k: int = 10

    def __post_init__(self) -> None:
        if self.mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of: {', '.join(RETRIEVAL_MODES)}")
        if self.chunk_tokens < 32:
            raise ValueError("chunk_tokens must be at least 32")
        if self.candidate_pool < self.top_k:
            raise ValueError("candidate_pool must be at least top_k")
        if self.retrieval_depth < self.candidate_pool:
            raise ValueError("retrieval_depth must be at least candidate_pool")

    @property
    def uses_dense(self) -> bool:
        return self.mode in ("dense", "hybrid", "routed")

    @property
    def uses_lexical(self) -> bool:
        return self.mode in ("lexical", "hybrid", "routed")

    def label(self) -> str:
        """A stable, readable identifier used as the row key in report tables."""

        model = self.embedding_model.rsplit("/", 1)[-1] if self.uses_dense else "none"
        return (
            f"chunk{self.chunk_tokens}"
            f"|{self.mode}"
            f"|{model}"
            f"|rerank={'on' if self.rerank else 'off'}"
        )

    def index_label(self) -> str:
        """Identifier for artefacts shared across configs: the chunk index."""

        return f"chunk{self.chunk_tokens}_overlap{int(self.overlap_ratio * 100)}"

    def with_(self, **changes) -> RetrievalConfig:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label(),
            "chunk_tokens": self.chunk_tokens,
            "overlap_ratio": self.overlap_ratio,
            "mode": self.mode,
            "embedding_model": self.embedding_model if self.uses_dense else "",
            "rrf_k": self.rrf_k,
            "rerank": self.rerank,
            "rerank_model": self.rerank_model if self.rerank else "",
            "retrieval_depth": self.retrieval_depth,
            "candidate_pool": self.candidate_pool,
            "top_k": self.top_k,
        }
