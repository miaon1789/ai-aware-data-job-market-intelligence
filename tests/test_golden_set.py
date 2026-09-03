import json
from pathlib import Path

import pandas as pd
import pytest

from job_market_intelligence.retrieval.golden import (
    condense,
    document_hash,
    judged_documents,
    load_queries,
    resolve_relevance,
)

ROOT = Path(__file__).resolve().parents[1]

JOBS = pd.DataFrame(
    [
        {"job_id": "a", "title": "Data Engineer", "description": "we use dbt daily",
         "description_structured": "we use dbt daily", "city": "Sydney",
         "analysis_role": "Data Engineer", "source": "greenhouse-ats"},
        {"job_id": "b", "title": "Analyst", "description": "Power BI reporting",
         "description_structured": "Power BI reporting", "city": "Melbourne",
         "analysis_role": "Data Analyst", "source": "adzuna-api"},
        {"job_id": "c", "title": "Scientist", "description": "no tools named here",
         "description_structured": "no tools named here", "city": "Sydney",
         "analysis_role": "Data Scientist", "source": "adzuna-api"},
    ]
)
SKILLS = pd.DataFrame(
    [
        {"job_id": "a", "skill": "SQL", "category": "Data", "context": "required"},
        {"job_id": "b", "skill": "Power BI", "category": "BI", "context": "mentioned"},
        {"job_id": "zz", "skill": "SQL", "category": "Data", "context": "required"},
    ]
)
LABELS = pd.DataFrame(
    [
        {"job_id": "a", "role_label": "Data Engineer", "entry_fit_label": "Experienced role"},
        {"job_id": "b", "role_label": "Data Analyst / BI",
         "entry_fit_label": "Likely entry-level"},
    ]
)


def _query(**overrides):
    payload = {
        "query_id": "q1", "query": "dbt", "query_class": "lexical",
        "relevance": {"type": "alias_match", "aliases": ["dbt"]},
    }
    payload.update(overrides)
    return load_queries_from_payloads([payload])[0]


def load_queries_from_payloads(payloads, tmp_path=None):
    import tempfile

    directory = tmp_path or Path(tempfile.mkdtemp())
    path = Path(directory) / "queries.jsonl"
    path.write_text("\n".join(json.dumps(p) for p in payloads) + "\n", encoding="utf-8")
    return list(load_queries(path))


def test_document_hash_is_stable_and_not_the_job_id():
    assert document_hash("5771223129") == document_hash("5771223129")
    assert document_hash("5771223129") != document_hash("5771223130")
    assert len(document_hash("x")) == 20
    assert "5771223129" not in document_hash("5771223129")


def test_alias_match_respects_word_boundaries():
    query = _query(relevance={"type": "alias_match", "aliases": ["dbt"]})
    assert resolve_relevance(query, jobs=JOBS) == {"a"}
    # "bi" must not match inside "Bird" or as part of another token.
    boundary = _query(relevance={"type": "alias_match", "aliases": ["power bi"]})
    assert resolve_relevance(boundary, jobs=JOBS) == {"b"}


def test_alias_match_searches_the_title_as_well_as_the_body():
    query = _query(relevance={"type": "alias_match", "aliases": ["scientist"]})
    assert resolve_relevance(query, jobs=JOBS) == {"c"}


def test_skill_match_ignores_advertisements_outside_the_corpus():
    query = _query(relevance={"type": "skill_match", "skills": ["SQL"]})
    # "zz" has a skill mention but is not in the corpus frame.
    assert resolve_relevance(query, jobs=JOBS, skill_mentions=SKILLS) == {"a"}


def test_skill_match_can_narrow_to_a_context():
    query = _query(relevance={"type": "skill_match", "skills": ["SQL"],
                              "contexts": ["preferred"]})
    assert resolve_relevance(query, jobs=JOBS, skill_mentions=SKILLS) == set()


def test_label_match_selects_by_adjudicated_label():
    query = _query(relevance={"type": "label_match", "column": "entry_fit_label",
                              "values": ["Likely entry-level"]})
    assert resolve_relevance(query, jobs=JOBS, document_labels=LABELS) == {"b"}


def test_metadata_match_selects_by_corpus_column():
    query = _query(relevance={"type": "metadata_match", "column": "city",
                              "values": ["Sydney"]})
    assert resolve_relevance(query, jobs=JOBS) == {"a", "c"}


def test_rules_raise_when_their_inputs_are_missing():
    with pytest.raises(ValueError):
        resolve_relevance(
            _query(relevance={"type": "skill_match", "skills": ["SQL"]}), jobs=JOBS
        )
    with pytest.raises(ValueError):
        resolve_relevance(
            _query(relevance={"type": "label_match", "column": "nope", "values": ["x"]}),
            jobs=JOBS, document_labels=LABELS,
        )


def test_judged_pool_is_the_whole_corpus_or_the_labelled_subset():
    assert judged_documents(_query(), jobs=JOBS) is None
    labelled = judged_documents(
        _query(judged_pool="labelled"), jobs=JOBS, document_labels=LABELS
    )
    assert labelled == {"a", "b"}


def test_condense_removes_unjudged_documents_without_reordering():
    assert condense(["c", "a", "b"], {"a", "b"}) == ["a", "b"]
    assert condense(["c", "a"], None) == ["c", "a"]


def test_loader_rejects_duplicate_ids_and_unknown_rule_types(tmp_path):
    with pytest.raises(ValueError):
        load_queries_from_payloads(
            [
                {"query_id": "q1", "query": "a", "query_class": "lexical",
                 "relevance": {"type": "alias_match", "aliases": ["a"]}},
                {"query_id": "q1", "query": "b", "query_class": "lexical",
                 "relevance": {"type": "alias_match", "aliases": ["b"]}},
            ],
            tmp_path,
        )
    with pytest.raises(ValueError):
        load_queries_from_payloads(
            [{"query_id": "q2", "query": "a", "query_class": "lexical",
              "relevance": {"type": "vibes", "aliases": ["a"]}}],
            tmp_path,
        )


def test_committed_golden_set_is_valid_and_balanced():
    golden = load_queries(ROOT / "eval/golden/queries.jsonl")
    assert len(golden) >= 50
    for query_class in ("lexical", "semantic", "aggregation"):
        assert len(golden.by_class(query_class)) >= 10


def test_committed_relevance_file_carries_no_recoverable_job_ids():
    """The published evaluation artefact must contain hashes, not ids."""

    path = ROOT / "eval/golden/relevance_hashed.jsonl"
    if not path.exists():  # generated by scripts/build_golden_set.py
        pytest.skip("relevance_hashed.jsonl has not been generated")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assert "relevant_job_ids" not in row
        for digest in row["relevant_document_hashes"]:
            assert len(digest) == 20
            assert all(character in "0123456789abcdef" for character in digest)
