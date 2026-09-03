"""Cross-encoder reranking of a coarse candidate pool.

Bi-encoder retrieval scores a query and a chunk independently, so it cannot see
term interactions. A cross-encoder reads the pair jointly and is markedly more
accurate -- and markedly slower, which is why it is applied to a shortlist
rather than the corpus. ``scripts/measure_retrieval_cost.py`` quantifies that
trade.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Pinned for the same reason as the embedding models; see embedding.py.
RERANK_REVISIONS = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
}


class CrossEncoderReranker:
    """Scores (query, chunk) pairs jointly and reorders the shortlist."""

    def __init__(
        self, model_name: str = DEFAULT_RERANK_MODEL, *, device: str | None = None
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                revision=RERANK_REVISIONS.get(self.model_name),
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        *,
        top_k: int = 10,
        batch_size: int = 32,
    ) -> list[tuple[str, float]]:
        """Reorder ``(identifier, text)`` pairs by joint relevance to ``query``."""

        if not candidates or top_k <= 0:
            return []
        scores = self.model.predict(
            [(str(query), text) for _, text in candidates],
            batch_size=batch_size,
            show_progress_bar=False,
        )
        scored = [
            (identifier, float(score))
            for (identifier, _), score in zip(candidates, scores, strict=True)
        ]
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]
