"""Reciprocal rank fusion of independently ranked result lists."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked identifier lists into one ranking.

    RRF scores by rank position rather than by score value, which is what makes
    it usable here: BM25 scores and cosine similarities are not comparable
    quantities and normalising them against each other would require a
    calibration set the corpus is too small to spare.

    ``k`` damps the contribution of top ranks; the conventional default of 60
    is used unless an ablation says otherwise.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[str, float] = {}
    for name, ranked in rankings.items():
        weight = 1.0 if weights is None else float(weights.get(name, 1.0))
        if weight == 0.0:
            continue
        for position, identifier in enumerate(ranked, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + weight / (k + position)

    # Ties broken by identifier so that runs are reproducible.
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
