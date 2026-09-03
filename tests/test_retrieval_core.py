import numpy as np
import pandas as pd
import pytest

from job_market_intelligence.retrieval import dense
from job_market_intelligence.retrieval.config import RetrievalConfig
from job_market_intelligence.retrieval.embedding import EmbeddingCache, cache_key, normalise
from job_market_intelligence.retrieval.fusion import reciprocal_rank_fusion
from job_market_intelligence.retrieval.lexical import LexicalIndex, tokenise_query


def test_rrf_ranks_a_consensus_document_above_either_leader():
    # "b" is second on both lists; "a" and "c" each lead exactly one and are
    # absent from the other. Agreement across branches outweighs one first place.
    fused = reciprocal_rank_fusion({"dense": ["a", "b"], "lexical": ["c", "b"]}, k=60)
    assert fused[0][0] == "b"
    assert dict(fused)["b"] > dict(fused)["a"]


def test_rrf_finds_no_separation_when_branches_disagree_symmetrically():
    # Each branch reverses the other, so no document has real support. The
    # scores must stay within a hair of each other rather than manufacturing a
    # winner -- the spread here is under a thousandth of the score itself.
    fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"], "lexical": ["c", "b", "a"]}, k=60)
    scores = [score for _, score in fused]
    assert max(scores) - min(scores) < 0.001 * max(scores)


def test_rrf_scores_by_rank_not_by_incomparable_scores():
    fused = dict(reciprocal_rank_fusion({"dense": ["x"], "lexical": ["x"]}, k=60))
    assert fused["x"] == pytest.approx(2 / 61)


def test_rrf_weights_can_silence_a_branch():
    fused = reciprocal_rank_fusion(
        {"dense": ["a"], "lexical": ["b"]}, k=60, weights={"dense": 0.0}
    )
    assert [identifier for identifier, _ in fused] == ["b"]


def test_rrf_is_deterministic_on_ties():
    rankings = {"one": ["b", "a"], "two": ["a", "b"]}
    assert reciprocal_rank_fusion(rankings) == reciprocal_rank_fusion(rankings)


def test_rrf_rejects_non_positive_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"dense": ["a"]}, k=0)


def test_dense_search_returns_sorted_cosine_neighbours():
    matrix = normalise(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32))
    hits = dense.search(np.array([1.0, 0.0], dtype=np.float32), matrix, top_k=2)
    assert [row for row, _ in hits] == [0, 1]
    assert hits[0][1] >= hits[1][1]


def test_dense_search_honours_the_allowed_row_filter():
    matrix = normalise(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32))
    hits = dense.search(
        np.array([1.0, 0.0], dtype=np.float32), matrix, top_k=2, allowed_rows=np.array([1, 2])
    )
    assert [row for row, _ in hits] == [1, 2]


def test_dense_search_rejects_a_dimension_mismatch():
    matrix = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        dense.search(np.zeros(2, dtype=np.float32), matrix, top_k=1)


def test_dense_search_handles_empty_inputs():
    assert dense.search(np.zeros(2, dtype=np.float32), np.zeros((0, 2), dtype=np.float32)) == []


def test_normalise_leaves_zero_rows_finite():
    result = normalise(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))
    assert np.isfinite(result).all()
    assert result[1] == pytest.approx([0.6, 0.8])


def test_embedding_cache_round_trips_through_disk(tmp_path):
    cache = EmbeddingCache.load(tmp_path, "model-a")
    cache.put(["hello"], np.array([[1.0, 2.0]], dtype=np.float32))
    cache.save()

    reloaded = EmbeddingCache.load(tmp_path, "model-a")
    assert reloaded.get("hello") == pytest.approx([1.0, 2.0])
    assert reloaded.get("missing") is None


def test_embedding_cache_keys_are_model_specific():
    assert cache_key("model-a", "text") != cache_key("model-b", "text")


def test_lexical_index_matches_a_rare_token(tmp_path):
    chunks = pd.DataFrame(
        [
            {"chunk_id": "c1", "job_id": "j1", "text": "We build dbt models and Airflow DAGs",
             "city": "Sydney", "analysis_role": "Data Engineer", "seniority": "Unclear",
             "source": "greenhouse-ats", "section": "requirements"},
            {"chunk_id": "c2", "job_id": "j2", "text": "Dashboards in Power BI for retail",
             "city": "Melbourne", "analysis_role": "Data Analyst", "seniority": "Unclear",
             "source": "adzuna-api", "section": "intro"},
        ]
    )
    index = LexicalIndex.build(chunks, tmp_path / "lex.duckdb")

    hits = index.search("dbt", top_k=5)
    assert [chunk_id for chunk_id, _ in hits] == ["c1"]
    assert index.search("dashboards", top_k=5)[0][0] == "c2"


def test_lexical_index_applies_metadata_filters(tmp_path):
    chunks = pd.DataFrame(
        [
            {"chunk_id": "c1", "job_id": "j1", "text": "python engineering role",
             "city": "Sydney", "analysis_role": "Data Engineer", "seniority": "Unclear",
             "source": "greenhouse-ats", "section": "requirements"},
            {"chunk_id": "c2", "job_id": "j2", "text": "python analytics role",
             "city": "Melbourne", "analysis_role": "Data Analyst", "seniority": "Unclear",
             "source": "adzuna-api", "section": "requirements"},
        ]
    )
    index = LexicalIndex.build(chunks, tmp_path / "lex.duckdb")

    hits = index.search("python", top_k=5, filters={"city": "Melbourne"})
    assert [chunk_id for chunk_id, _ in hits] == ["c2"]
    with pytest.raises(ValueError):
        index.search("python", filters={"salary": "high"})


def test_lexical_index_requires_text(tmp_path):
    with pytest.raises(ValueError):
        LexicalIndex.build(pd.DataFrame([{"chunk_id": "c1"}]), tmp_path / "lex.duckdb")


def test_tokenise_query_keeps_technical_tokens():
    assert tokenise_query("scikit-learn & C#") == "scikit learn c#"
    assert tokenise_query("") == ""


def test_config_labels_are_stable_and_distinguish_axes():
    baseline = RetrievalConfig()
    assert baseline.label() == "chunk512|hybrid|bge-small-en-v1.5|rerank=off"
    assert baseline.with_(rerank=True).label().endswith("rerank=on")
    assert baseline.with_(mode="lexical").label() == "chunk512|lexical|none|rerank=off"
    assert baseline.index_label() == "chunk512_overlap10"


def test_config_rejects_incoherent_settings():
    with pytest.raises(ValueError):
        RetrievalConfig(mode="fuzzy")
    with pytest.raises(ValueError):
        RetrievalConfig(candidate_pool=5, top_k=10)
    with pytest.raises(ValueError):
        RetrievalConfig(retrieval_depth=10, candidate_pool=50, top_k=10)


def test_lexical_mode_does_not_claim_to_use_an_embedding_model():
    config = RetrievalConfig(mode="lexical")
    assert not config.uses_dense
    assert config.uses_lexical
    assert config.to_dict()["embedding_model"] == ""


def test_routed_mode_needs_both_indexes():
    config = RetrievalConfig(mode="routed")
    assert config.uses_dense and config.uses_lexical
    assert config.label() == "chunk512|routed|bge-small-en-v1.5|rerank=off"


def test_routed_mode_dispatches_by_query_shape(tmp_path):
    """Routing runs one branch, and which one depends on the query."""

    import numpy as np

    from job_market_intelligence.retrieval.search import HybridRetriever

    chunks = pd.DataFrame(
        [
            {"chunk_id": "c1", "job_id": "j1", "text": "we use dbt every day",
             "body": "we use dbt every day", "title": "Data Engineer", "section": "requirements",
             "heading": "", "city": "Sydney", "company": "N", "posted_at": "2026-05-01",
             "source": "greenhouse-ats", "analysis_role": "Data Engineer",
             "seniority": "Unclear"},
            {"chunk_id": "c2", "job_id": "j2", "text": "explain results to stakeholders",
             "body": "explain results to stakeholders", "title": "Analyst",
             "section": "requirements", "heading": "", "city": "Sydney", "company": "H",
             "posted_at": "2026-05-02", "source": "adzuna-api",
             "analysis_role": "Data Analyst", "seniority": "Unclear"},
        ]
    )
    index = LexicalIndex.build(chunks, tmp_path / "lex.duckdb")
    # Orthogonal vectors: the dense branch always prefers c2.
    embeddings = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    class StubEncoder:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    retriever = HybridRetriever(
        chunks,
        config=RetrievalConfig(mode="routed", retrieval_depth=10, candidate_pool=5, top_k=5),
        embeddings=embeddings,
        lexical_index=index,
        encoder=StubEncoder(),
    )

    assert retriever._active_branches("dbt") == ("lexical",)
    assert retriever._active_branches("roles that involve explaining things") == ("dense",)
    assert set(retriever._active_branches("how many roles mention dbt?")) == {"dense", "lexical"}

    # A lexical query takes the BM25 branch and finds the advertisement that
    # actually contains the term, not the one the stub embedding prefers.
    assert retriever.search("dbt")[0].job_id == "j1"
    # A semantic query takes the dense branch, which prefers c2 by construction.
    assert retriever.search("roles that involve explaining things")[0].job_id == "j2"
