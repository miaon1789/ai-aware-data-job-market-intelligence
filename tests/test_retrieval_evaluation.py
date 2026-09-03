import numpy as np
import pandas as pd
import pytest

from job_market_intelligence.retrieval.evaluation import (
    aggregate_scores,
    bootstrap_ci,
    dedupe_to_documents,
    ndcg_at_k,
    paired_comparison,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    resolution_limit,
    routing_break_even,
    score_query,
)


def test_dedupe_collapses_chunks_of_one_advertisement_to_one_hit():
    chunk_to_job = {"a::00": "a", "a::01": "a", "b::00": "b"}
    assert dedupe_to_documents(["a::00", "a::01", "b::00"], chunk_to_job) == ["a", "b"]


def test_dedupe_keeps_the_first_position_of_each_advertisement():
    chunk_to_job = {"a::00": "a", "b::00": "b", "a::01": "a"}
    assert dedupe_to_documents(["b::00", "a::00", "a::01"], chunk_to_job) == ["b", "a"]


def test_dedupe_ignores_chunks_with_no_known_advertisement():
    assert dedupe_to_documents(["ghost::00"], {"a::00": "a"}) == []


def test_recall_and_precision_at_k():
    relevant = {"a", "b", "c"}
    ranked = ["a", "x", "b", "y", "z"]
    assert recall_at_k(relevant, ranked, 5) == pytest.approx(2 / 3)
    assert precision_at_k(relevant, ranked, 5) == pytest.approx(2 / 5)
    assert recall_at_k(relevant, ranked, 1) == pytest.approx(1 / 3)


def test_recall_is_undefined_without_judgements():
    assert np.isnan(recall_at_k(set(), ["a"], 5))


def test_reciprocal_rank_finds_the_first_hit():
    assert reciprocal_rank({"c"}, ["a", "b", "c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank({"z"}, ["a", "b"]) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking_and_normalised_for_set_size():
    assert ndcg_at_k({"a", "b"}, ["a", "b", "c"], 3) == pytest.approx(1.0)
    # A single relevant document ranked first is also perfect: the ideal
    # discount is capped at min(k, |relevant|), so nDCG does not punish a
    # small relevant set the way recall@k does.
    assert ndcg_at_k({"a"}, ["a", "b", "c"], 3) == pytest.approx(1.0)
    assert ndcg_at_k({"a"}, ["b", "a"], 2) < 1.0


def test_score_query_reports_every_metric():
    scores = score_query({"a"}, ["a", "b"], ks=(1, 2))
    assert set(scores) == {"mrr", "recall@1", "precision@1", "ndcg@1",
                           "recall@2", "precision@2", "ndcg@2"}


def test_bootstrap_interval_brackets_the_mean_and_is_reproducible():
    values = [0.2, 0.4, 0.6, 0.8, 1.0]
    low, high = bootstrap_ci(values, iterations=500, seed=7)
    assert low <= np.mean(values) <= high
    assert bootstrap_ci(values, iterations=500, seed=7) == (low, high)


def test_bootstrap_interval_widens_as_the_sample_shrinks():
    wide = bootstrap_ci([0.0, 1.0], iterations=500, seed=1)
    narrow = bootstrap_ci([0.5] * 40, iterations=500, seed=1)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_bootstrap_handles_degenerate_inputs():
    assert np.isnan(bootstrap_ci([])).all()
    assert bootstrap_ci([0.5]) == (0.5, 0.5)


def _per_query_frame():
    return pd.DataFrame(
        [
            {"query_class": "lexical", "recall@10": 0.9, "precision@5": 0.8, "mrr": 1.0},
            {"query_class": "lexical", "recall@10": 0.7, "precision@5": 0.6, "mrr": 0.5},
            {"query_class": "semantic", "recall@10": 0.2, "precision@5": 0.1, "mrr": 0.2},
            {"query_class": "semantic", "recall@10": 0.3, "precision@5": 0.2, "mrr": 0.3},
        ]
    )


def test_aggregate_scores_reports_means_with_intervals():
    result = aggregate_scores(_per_query_frame(), iterations=200)
    assert len(result) == 1
    assert result.loc[0, "queries"] == 4
    assert result.loc[0, "recall@10"] == pytest.approx(0.525)
    assert result.loc[0, "recall@10_lo"] <= result.loc[0, "recall@10"]
    assert result.loc[0, "recall@10_hi"] >= result.loc[0, "recall@10"]


def test_aggregate_scores_splits_by_query_class():
    result = aggregate_scores(
        _per_query_frame(), group_columns=["query_class"], iterations=200
    )
    lexical = result[result.query_class.eq("lexical")].iloc[0]
    semantic = result[result.query_class.eq("semantic")].iloc[0]
    # The overall mean (0.525) describes neither class.
    assert lexical["recall@10"] == pytest.approx(0.8)
    assert semantic["recall@10"] == pytest.approx(0.25)


def test_aggregate_scores_rejects_a_missing_group_column():
    with pytest.raises(ValueError):
        aggregate_scores(_per_query_frame(), group_columns=["source"])


def test_aggregate_scores_requires_metric_columns():
    with pytest.raises(ValueError):
        aggregate_scores(pd.DataFrame([{"query_class": "lexical"}]))


def test_resolution_limit_is_the_interval_half_width():
    frame = _per_query_frame()
    limit = resolution_limit(frame, "recall@10", iterations=500)
    low, high = bootstrap_ci(frame["recall@10"].to_numpy(), iterations=500)
    assert limit == pytest.approx((high - low) / 2, abs=1e-4)
    with pytest.raises(ValueError):
        resolution_limit(frame, "ndcg@99")


def _two_config_frame():
    """Config A is much better on lexical, much worse on semantic."""

    rows = []
    for index in range(6):
        lexical = index < 3
        rows.append(
            {
                "config": "A", "query_id": f"q{index}",
                "query_class": "lexical" if lexical else "semantic",
                "ndcg@10": 0.9 if lexical else 0.1, "recall@10": 0.9 if lexical else 0.1,
            }
        )
        rows.append(
            {
                "config": "B", "query_id": f"q{index}",
                "query_class": "lexical" if lexical else "semantic",
                "ndcg@10": 0.5 if lexical else 0.5, "recall@10": 0.5 if lexical else 0.5,
            }
        )
    return pd.DataFrame(rows)


def test_paired_comparison_reports_direction_wins_and_interval():
    result = paired_comparison(_two_config_frame(), "A", "B", iterations=500)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["queries"] == 6
    assert row["mean_difference"] == pytest.approx(0.0)
    assert row["a_wins"] == 3
    assert row["a_loses"] == 3
    assert row["ties"] == 0


def test_paired_comparison_by_class_finds_what_the_overall_mean_hides():
    result = paired_comparison(
        _two_config_frame(), "A", "B", group_columns=["query_class"], iterations=500
    )
    lexical = result[result.query_class.eq("lexical")].iloc[0]
    semantic = result[result.query_class.eq("semantic")].iloc[0]
    # The overall difference is exactly zero; per class it is +0.4 and -0.4.
    assert lexical["mean_difference"] == pytest.approx(0.4)
    assert semantic["mean_difference"] == pytest.approx(-0.4)
    assert lexical["distinguishable"] and semantic["distinguishable"]


def test_paired_comparison_marks_an_interval_spanning_zero_as_indistinguishable():
    frame = pd.DataFrame(
        [
            {"config": c, "query_id": f"q{i}", "query_class": "lexical",
             "ndcg@10": 0.5 + (0.01 if c == "A" else 0.0) * (1 if i % 2 else -1)}
            for i in range(10) for c in ("A", "B")
        ]
    )
    assert not paired_comparison(frame, "A", "B", iterations=500).iloc[0]["distinguishable"]


def test_paired_comparison_rejects_unknown_configs_and_metrics():
    frame = _two_config_frame()
    with pytest.raises(ValueError):
        paired_comparison(frame, "A", "missing")
    with pytest.raises(ValueError):
        paired_comparison(frame, "A", "B", metric="map@10")


def test_routing_break_even_is_between_zero_and_one_and_ordered():
    frame = _two_config_frame()
    # Routing sends lexical queries to A and semantic queries to B; B is also
    # the fusion baseline here.
    result = routing_break_even(
        frame,
        branch_for_class={"lexical": "A", "semantic": "B"},
        fallback_config="B",
        fusion_config="B",
    )
    assert result["routed_perfect_router"] > result["fusion"]
    assert result["routed_always_wrong"] < result["fusion"]
    assert 0.0 < result["break_even_router_accuracy"] < 1.0


def test_routing_break_even_demands_a_perfect_router_when_routing_cannot_help():
    """If routing has nothing to gain, no achievable accuracy justifies it."""

    frame = pd.DataFrame(
        [
            {"config": c, "query_id": f"q{i}", "query_class": "lexical", "ndcg@10": 0.5}
            for i in range(4) for c in ("A", "B")
        ]
    )
    result = routing_break_even(
        frame, branch_for_class={"lexical": "A"}, fallback_config="B", fusion_config="B"
    )
    assert np.isnan(result["break_even_router_accuracy"])
