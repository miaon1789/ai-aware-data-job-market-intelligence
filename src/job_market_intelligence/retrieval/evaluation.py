"""Retrieval metrics with bootstrap confidence intervals.

Relevance is judged at *advertisement* level, not chunk level: a query asking
for roles requiring dbt is answered by an advertisement, and returning three
chunks of the same advertisement is one hit, not three. Rankings are therefore
collapsed to their first occurrence of each ``job_id`` before scoring.

Every aggregate is reported with a bootstrap interval over queries. Differences
between configurations require a paired interval. An interval containing zero
does not establish equivalence or prove the absence of regression.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_KS = (1, 5, 10)


def dedupe_to_documents(ranked_chunk_ids: Sequence[str], chunk_to_job: dict[str, str]) -> list[str]:
    """Collapse a chunk ranking to the advertisement ranking it implies."""

    seen: set[str] = set()
    documents: list[str] = []
    for chunk_id in ranked_chunk_ids:
        job_id = chunk_to_job.get(chunk_id)
        if job_id is None or job_id in seen:
            continue
        seen.add(job_id)
        documents.append(job_id)
    return documents


def recall_at_k(relevant: set[str], ranked: Sequence[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(relevant & set(ranked[:k])) / len(relevant)


def precision_at_k(relevant: set[str], ranked: Sequence[str], k: int) -> float:
    if k <= 0:
        return float("nan")
    return len(relevant & set(ranked[:k])) / k


def reciprocal_rank(relevant: set[str], ranked: Sequence[str]) -> float:
    for position, identifier in enumerate(ranked, start=1):
        if identifier in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(relevant: set[str], ranked: Sequence[str], k: int) -> float:
    """Binary-gain nDCG: every relevant advertisement counts equally."""

    if not relevant:
        return float("nan")
    gains = [1.0 if identifier in relevant else 0.0 for identifier in ranked[:k]]
    discounted = sum(gain / np.log2(position + 1) for position, gain in enumerate(gains, start=1))
    ideal = sum(
        1.0 / np.log2(position + 1) for position in range(1, min(k, len(relevant)) + 1)
    )
    return float(discounted / ideal) if ideal else float("nan")


def score_query(
    relevant: set[str], ranked_documents: Sequence[str], *, ks: Iterable[int] = DEFAULT_KS
) -> dict[str, float]:
    """Compute every metric for a single query."""

    scores: dict[str, float] = {"mrr": reciprocal_rank(relevant, ranked_documents)}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(relevant, ranked_documents, k)
        scores[f"precision@{k}"] = precision_at_k(relevant, ranked_documents, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(relevant, ranked_documents, k)
    return scores


def bootstrap_ci(
    values: Sequence[float],
    *,
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean, resampling over queries."""

    clean = np.asarray([value for value in values if not np.isnan(value)], dtype=float)
    if clean.size == 0:
        return (float("nan"), float("nan"))
    if clean.size == 1:
        return (float(clean[0]), float(clean[0]))
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, clean.size, size=(iterations, clean.size))
    means = clean[draws].mean(axis=1)
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lower, upper)


def metric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith(("recall@", "precision@", "ndcg@")) or column == "mrr"
    ]


def aggregate_scores(
    per_query: pd.DataFrame,
    *,
    group_columns: Sequence[str] = (),
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Average per-query scores, attaching a bootstrap interval to each mean.

    Passing ``group_columns=["query_class"]`` produces the by-class breakdown
    that the headline result depends on; an overall mean alone would hide a
    gain that lands entirely in one class.
    """

    metrics = metric_columns(per_query)
    if not metrics:
        raise ValueError("per-query frame contains no metric columns")

    groups: list[tuple[dict[str, object], pd.DataFrame]]
    if group_columns:
        missing = sorted(set(group_columns) - set(per_query.columns))
        if missing:
            raise ValueError(f"per-query frame is missing columns: {', '.join(missing)}")
        groups = []
        for keys, part in per_query.groupby(list(group_columns), dropna=False):
            values = keys if isinstance(keys, tuple) else (keys,)
            groups.append((dict(zip(group_columns, values, strict=True)), part))
    else:
        groups = [({}, per_query)]

    rows: list[dict[str, object]] = []
    for keys, part in groups:
        row: dict[str, object] = dict(keys)
        row["queries"] = int(len(part))
        for metric in metrics:
            values = part[metric].to_numpy(dtype=float)
            mean = float(np.nanmean(values)) if len(values) else float("nan")
            lower, upper = bootstrap_ci(values, iterations=iterations, alpha=alpha, seed=seed)
            row[metric] = round(mean, 4)
            row[f"{metric}_lo"] = round(lower, 4)
            row[f"{metric}_hi"] = round(upper, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def resolution_limit(per_query: pd.DataFrame, metric: str = "recall@10", **kwargs) -> float:
    """Single-configuration mean interval half-width, not a detection threshold."""

    if metric not in per_query.columns:
        raise ValueError(f"metric not present: {metric}")
    lower, upper = bootstrap_ci(per_query[metric].to_numpy(dtype=float), **kwargs)
    return round((upper - lower) / 2, 4)


def paired_comparison(
    per_query: pd.DataFrame,
    config_a: str,
    config_b: str,
    *,
    metric: str = "ndcg@10",
    group_columns: Sequence[str] = (),
    iterations: int = 20000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare two configurations on identical, unique query sets.

    Two overlapping confidence intervals do not mean two systems are
    indistinguishable: the configurations were run on the *same* queries, so
    the difference can be bootstrapped directly. Pairing removes between-query
    variance -- which dominates here, because a query with three relevant
    advertisements and one with eighty score nothing alike -- and detects
    differences that separate per-system intervals cannot.

    Win/loss/tie counts are reported beside the mean because they answer a
    different question: not "how much better on average" but "on how many
    queries is it ever worse".
    """

    required = {"config", "query_id", metric, *group_columns}
    if not required.issubset(per_query.columns):
        raise ValueError(f"missing comparison columns: {sorted(required - set(per_query.columns))}")
    if config_a == config_b:
        raise ValueError("comparison requires distinct configuration identities")
    frames = {}
    for config in (config_a, config_b):
        part = per_query[per_query["config"].eq(config)]
        if part.empty:
            raise ValueError(f"configuration not present: {config}")
        if part["query_id"].isna().any() or part["query_id"].duplicated().any():
            raise ValueError("query ids must be present and unique per configuration")
        if not np.isfinite(part[metric].to_numpy(dtype=float)).all():
            raise ValueError("comparison scores must be finite")
        frames[config] = part.set_index("query_id")

    shared = frames[config_a].index
    if set(shared) != set(frames[config_b].index):
        raise ValueError("the two configurations must contain identical query ids")
    if group_columns:
        left_labels = frames[config_a].loc[shared, list(group_columns)]
        right_labels = frames[config_b].loc[shared, list(group_columns)]
        if left_labels.isna().any().any() or not left_labels.equals(right_labels):
            raise ValueError("query group labels must be present and identical")

    groups: list[tuple[dict[str, object], pd.Index]] = [({}, shared)]
    if group_columns:
        labels = frames[config_a].loc[shared, list(group_columns)]
        groups = []
        for keys, part in labels.groupby(list(group_columns), dropna=False):
            values = keys if isinstance(keys, tuple) else (keys,)
            groups.append((dict(zip(group_columns, values, strict=True)), part.index))

    rows: list[dict[str, object]] = []
    for keys, index in groups:
        left = frames[config_a].loc[index, metric].to_numpy(dtype=float)
        right = frames[config_b].loc[index, metric].to_numpy(dtype=float)
        difference = left - right
        low, high = bootstrap_ci(difference, iterations=iterations, alpha=alpha, seed=seed)
        rows.append(
            {
                **keys,
                "metric": metric,
                "config_a": config_a,
                "config_b": config_b,
                "queries": int(len(index)),
                "mean_a": float(np.mean(left)),
                "mean_b": float(np.mean(right)),
                "mean_difference": float(np.mean(difference)),
                "difference_lo": low,
                "difference_hi": high,
                # An interval excluding zero is the only case where the two
                # configurations are actually told apart by this evaluation set.
                "distinguishable": bool(low > 0 or high < 0),
                "a_wins": int((difference > 0).sum()),
                "a_loses": int((difference < 0).sum()),
                "ties": int((difference == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def routing_break_even(
    per_query: pd.DataFrame,
    *,
    branch_for_class: Mapping[str, str],
    fallback_config: str,
    fusion_config: str,
    metric: str = "ndcg@10",
) -> dict[str, float]:
    """The router accuracy at which routing stops beating fusion.

    Routing to a single branch beats fusing both only while the classifier is
    right often enough. Model a misrouted query as receiving the other
    branch's score and solve for the accuracy at which expected routed
    performance equals fusion. Reporting this number is what turns "routing
    wins" into a claim with a stated condition attached.
    """

    scores = per_query.pivot_table(index="query_id", columns="config", values=metric)
    classes = per_query.drop_duplicates("query_id").set_index("query_id")["query_class"]
    classes = classes.reindex(scores.index)

    correct, wrong = [], []
    for query_id, query_class in classes.items():
        branch = branch_for_class.get(query_class, fallback_config)
        alternatives = [
            config for config in branch_for_class.values() if config != branch
        ] or [fallback_config]
        correct.append(scores.at[query_id, branch])
        wrong.append(min(scores.at[query_id, config] for config in alternatives))

    correct_mean = float(np.nanmean(correct))
    wrong_mean = float(np.nanmean(wrong))
    fusion_mean = float(np.nanmean(scores[fusion_config].to_numpy(dtype=float)))
    spread = correct_mean - wrong_mean
    break_even = (fusion_mean - wrong_mean) / spread if spread else float("nan")
    return {
        "metric": metric,
        "routed_perfect_router": round(correct_mean, 4),
        "routed_always_wrong": round(wrong_mean, 4),
        "fusion": round(fusion_mean, 4),
        "break_even_router_accuracy": round(float(break_even), 4),
    }


def wilson_interval(successes: int, trials: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Used instead of the normal approximation because the counts that matter
    here are small and often perfect. A classifier that is right 27 times out
    of 27 has a point estimate of 1.0 and an interval that still reaches well
    below it; quoting the point estimate alone would overstate what a sample
    that size can establish.
    """

    if trials <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    spread = (
        z
        * float(np.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2)))
        / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))
