import json

import pytest

from job_market_intelligence.llm_annotation import (
    build_llm_audit_messages,
    disagreement_dimensions,
    llm_audit_json_schema,
    parse_llm_audit_response,
    summarize_llm_audit_review,
)


def test_builds_prompt_with_observed_and_inferred_instructions():
    messages = build_llm_audit_messages(
        "Data Scientist\n\nDevelop machine learning models.",
        current_labels=["ROLE::Data Scientist / ML"],
    )

    assert messages[0]["role"] == "system"
    assert "visible in the excerpt" in messages[1]["content"]
    assert "coding_likely_by_role" in messages[1]["content"]
    assert "ROLE::Data Scientist / ML" in messages[1]["content"]


def test_json_schema_matches_expected_top_level_contract():
    schema = llm_audit_json_schema()

    assert schema["name"] == "job_label_audit"
    assert schema["schema"]["required"] == [
        "observed",
        "inferred",
        "confidence",
        "rationale",
    ]
    assert "Data Scientist / ML" in schema["schema"]["properties"]["observed"][
        "properties"
    ]["role"]["enum"]


def test_parses_valid_llm_audit_response():
    content = json.dumps(
        {
            "observed": {
                "role": "Data Scientist / ML",
                "entry_fit": "Unclear",
                "ai_signal": "Explicit AI / ML system work",
                "coding_signal": "No coding signal",
            },
            "inferred": {
                "coding_likely_by_role": "Likely coding-intensive",
            },
            "confidence": "medium",
            "rationale": "ML is visible, but coding tools are not visible.",
        }
    )

    audit = parse_llm_audit_response(content)

    assert audit.ai_signal == "Explicit AI / ML system work"
    assert audit.coding_signal == "No coding signal"
    assert audit.coding_likely_by_role == "Likely coding-intensive"


def test_rejects_invalid_llm_label():
    content = json.dumps(
        {
            "observed": {
                "role": "Data Wizard",
                "entry_fit": "Unclear",
                "ai_signal": "No AI signal",
                "coding_signal": "No coding signal",
            },
            "inferred": {"coding_likely_by_role": "Unclear"},
            "confidence": "low",
            "rationale": "Invalid role.",
        }
    )

    with pytest.raises(ValueError, match="invalid observed.role"):
        parse_llm_audit_response(content)


def test_reports_disagreement_dimensions():
    content = json.dumps(
        {
            "observed": {
                "role": "Other / Mixed",
                "entry_fit": "Experienced role",
                "ai_signal": "No AI signal",
                "coding_signal": "No coding signal",
            },
            "inferred": {"coding_likely_by_role": "Likely low/no coding"},
            "confidence": "high",
            "rationale": "Sales role with employer AI boilerplate.",
        }
    )
    audit = parse_llm_audit_response(content)

    disagreements = disagreement_dimensions(
        [
            "ROLE::Other / Mixed",
            "ENTRY_FIT::Experienced role",
            "AI_SIGNAL::Explicit GenAI / LLM tool",
            "CODING_SIGNAL::No coding signal",
        ],
        audit,
    )

    assert disagreements == ["AI_SIGNAL"]


def test_summarizes_llm_audit_review_without_text():
    summary = summarize_llm_audit_review(
        [
            {
                "confidence": "high",
                "disagreement_dimensions": "AI_SIGNAL | CODING_SIGNAL",
                "coding_likely_by_role": "Likely coding-intensive",
            },
            {
                "confidence": "medium",
                "disagreement_dimensions": "",
                "coding_likely_by_role": "Likely low/no coding",
            },
        ]
    )

    assert summary["rows"] == 2
    assert summary["rows_with_any_disagreement"] == 1
    assert summary["exact_agreement_share"] == 0.5
    assert summary["disagreement_by_dimension"] == {
        "AI_SIGNAL": 1,
        "CODING_SIGNAL": 1,
    }
