"""Brute-force dense retrieval.

**Why there is no vector database here.** The corpus is roughly 2,400 chunks.
A 2,400 x 384 float32 matrix is 3.7 MB; a full cosine scan is one BLAS
matrix-vector product and completes in well under a millisecond. An approximate
nearest-neighbour index would add a build step, a tuning surface and an
operational dependency in exchange for a speed-up on an operation that is
already far below the latency floor set by the embedding forward pass -- and it
would trade away exact recall to do it.

The interface below is the seam where that judgement would be revisited: past
roughly 100k chunks, `search` is the only function that needs to change to swap
in FAISS or pgvector.
"""

from __future__ import annotations

import numpy as np


def search(
    query_vector: np.ndarray,
    matrix: np.ndarray,
    *,
    top_k: int = 10,
    allowed_rows: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """Return ``(row_index, cosine_similarity)`` for the closest rows.

    Both inputs are assumed L2-normalised, so the dot product is the cosine.
    ``allowed_rows`` restricts the scan to a metadata-filtered subset.
    """

    if matrix.size == 0 or top_k <= 0:
        return []
    vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if vector.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"query dimension {vector.shape[0]} does not match index dimension "
            f"{matrix.shape[1]}"
        )

    if allowed_rows is None:
        scores = matrix @ vector
        candidate_rows = np.arange(len(matrix))
    else:
        candidate_rows = np.asarray(allowed_rows, dtype=np.int64)
        if candidate_rows.size == 0:
            return []
        scores = matrix[candidate_rows] @ vector

    limit = min(top_k, scores.shape[0])
    # argpartition finds the top-k boundary without sorting the whole corpus.
    top = np.argpartition(-scores, limit - 1)[:limit]
    top = top[np.argsort(-scores[top])]
    return [(int(candidate_rows[position]), float(scores[position])) for position in top]
