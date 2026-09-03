import json
from pathlib import Path

import pytest

from job_market_intelligence.rag.answer import (
    AnswerValidationError,
    check_citations,
    parse_answer,
    synthesise_answer,
)
from job_market_intelligence.rag.prompts import build_answer_prompt, format_passages
from job_market_intelligence.rag.routing import classify_query, route, routing_rationale

ROOT = Path(__file__).resolve().parents[1]
PASSAGES = [
    {"job_id": "gh-1", "title": "Data Engineer", "company": "Northwind",
     "heading": "Requirements", "body": "Three years with SQL and dbt."},
    {"job_id": "gh-2", "title": "Data Analyst", "company": "Harbour",
     "heading": "Responsibilities", "body": "Build dashboards for the retail team."},
]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many advertisements mention SQL?", "aggregation"),
        ("What share of roles are in Sydney?", "aggregation"),
        ("Which skill is most common?", "aggregation"),
        ("Compare Python against R.", "aggregation"),
        ("dbt", "lexical"),
        ("Power BI", "lexical"),
        ("roles that involve explaining findings to stakeholders", "semantic"),
        ("What do these ads say about on-call duties?", "semantic"),
    ],
)
def test_classify_query_separates_the_three_shapes(question, expected):
    assert classify_query(question) == expected


def test_counting_questions_route_away_from_retrieval():
    assert route("How many ads require SQL?") == "query_job_stats"
    assert route("dbt") == "search_job_ads"
    assert "SQL" in routing_rationale("How many ads require SQL?") or "count" in routing_rationale(
        "How many ads require SQL?"
    ).lower()


def test_empty_question_does_not_route_to_the_counting_tool():
    assert route("") == "search_job_ads"


def test_routing_holds_up_on_the_held_out_probe_set():
    """The published held-out accuracy must not silently regress.

    These probes were written after the routing patterns were frozen and were
    never used to tune them. The threshold is deliberately set at the measured
    value rather than at 100%: the heuristic misses some aggregation phrasings,
    and the prompt and tool descriptions are the second line of defence.
    """

    probes = [
        json.loads(line)
        for line in (ROOT / "eval/golden/routing_heldout.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    correct = sum(classify_query(p["question"]) == p["expected_class"] for p in probes)
    assert correct / len(probes) >= 0.83


def test_format_passages_numbers_and_attributes_every_passage():
    rendered = format_passages(PASSAGES)
    assert "[1] advertisement gh-1" in rendered
    assert "[2] advertisement gh-2" in rendered
    assert "section: Requirements" in rendered


def test_answer_prompt_carries_the_question_and_the_passages():
    prompt = build_answer_prompt("What is required?", PASSAGES)
    assert "What is required?" in prompt
    assert "Three years with SQL and dbt." in prompt


def test_parse_answer_accepts_a_well_formed_reply():
    payload = parse_answer('{"answerable": true, "answer": "Yes", "citations": ["gh-1"]}')
    assert payload["answerable"] is True


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        '["a", "list"]',
        '{"answer": "no answerable key", "citations": []}',
        '{"answerable": "yes", "answer": "x", "citations": []}',
        '{"answerable": true, "answer": "x", "citations": "gh-1"}',
    ],
)
def test_parse_answer_rejects_malformed_replies(reply):
    with pytest.raises(AnswerValidationError):
        parse_answer(reply)


def test_check_citations_flags_an_advertisement_that_was_never_supplied():
    assert check_citations(["gh-1", "gh-999"], PASSAGES) == ("gh-999",)
    assert check_citations(["gh-1"], PASSAGES) == ()


def test_synthesis_is_a_dry_run_unless_private_text_is_acknowledged():
    def client(system, user):  # pragma: no cover - must not be reached
        raise AssertionError("the client must not be called without acknowledgement")

    result = synthesise_answer("What is required?", PASSAGES, client=client)
    assert result.sent is False
    assert result.answerable is False
    assert "Dry run" in result.missing
    assert "Three years with SQL and dbt." in result.prompt


def test_synthesis_sends_and_validates_when_acknowledged():
    def client(system, user):
        return json.dumps(
            {"answerable": True, "answer": "SQL and dbt.", "citations": ["gh-1", "gh-404"],
             "missing": ""}
        )

    result = synthesise_answer(
        "What is required?", PASSAGES, client=client, acknowledge_private_text=True
    )
    assert result.sent is True
    assert result.answer == "SQL and dbt."
    assert result.unsupported_citations == ("gh-404",)
    assert result.supporting_job_ids == ("gh-1", "gh-2")


def test_synthesis_refuses_when_nothing_was_retrieved():
    result = synthesise_answer("anything", [], client=lambda s, u: "{}",
                               acknowledge_private_text=True)
    assert result.answerable is False
    assert "No passages" in result.missing


def test_reported_routing_accuracy_is_the_held_out_figure_not_a_pooled_one():
    """Guard against quoting a number that includes the tuning probes.

    An earlier README quoted 10/13 for aggregation detection, which pooled the
    held-out probes with the 7 the router's patterns were written against. The
    honest figure is held-out only, and it is much worse.
    """

    heldout = [
        json.loads(line)
        for line in (ROOT / "eval/golden/routing_heldout.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    aggregation = [p for p in heldout if p["expected_class"] == "aggregation"]
    correct = sum(classify_query(p["question"]) == "aggregation" for p in aggregation)
    assert len(aggregation) == 6
    assert correct == 3, (
        "held-out aggregation detection changed; update the README figures "
        "in 'What retrieval cannot answer' and 'Limitations' to match."
    )
