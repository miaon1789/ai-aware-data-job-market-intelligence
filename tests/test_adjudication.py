import pandas as pd
import pytest

from job_market_intelligence.adjudication import (
    apply_adjudications,
    build_adjudication_template,
    labels_to_dimensions,
)
from job_market_intelligence.annotation import task_hash


def test_labels_to_dimensions_parses_prefixed_label_string():
    labels = (
        "ROLE::Data Analyst / BI | ENTRY_FIT::Likely entry-level | "
        "AI_SIGNAL::No AI signal | CODING_SIGNAL::No coding signal"
    )

    result = labels_to_dimensions(labels)

    assert result == {
        "role_label": "Data Analyst / BI",
        "entry_fit_label": "Likely entry-level",
        "ai_signal_label": "No AI signal",
        "coding_signal_label": "No coding signal",
    }


def test_build_adjudication_template_blanks_disputed_dimensions_only():
    text = "Data Scientist\n\nBuild machine learning models."
    digest = task_hash(text)
    rows = build_adjudication_template(
        human_export_rows=[{"text": text}],
        disagreement_rows=[
            {
                "task_hash": digest,
                "job_id": "job-1",
                "title": "Data Scientist",
                "current_labels": (
                    "ROLE::Data Scientist / ML | ENTRY_FIT::Unclear | "
                    "AI_SIGNAL::AI-adjacent but vague | CODING_SIGNAL::No coding signal"
                ),
                "llm_observed_labels": (
                    "ROLE::Data Scientist / ML | ENTRY_FIT::Unclear | "
                    "AI_SIGNAL::Explicit AI / ML system work | "
                    "CODING_SIGNAL::No coding signal"
                ),
                "confidence": "high",
                "disagreement_dimensions": "AI_SIGNAL",
                "coding_likely_by_role": "Likely coding-intensive",
                "rationale": "ML is explicit.",
            }
        ],
        source_job_rows=[
            {
                "job_id": "job-1",
                "title": "Data Scientist",
                "company": "Example Co",
                "city": "Sydney",
                "posted_at": "2026-06-01",
                "collection_query": "data scientist",
                "collection_group": "entry_level",
                "source_url": "https://example.com/job",
            }
        ],
    )

    assert len(rows) == 1
    assert rows[0]["text"] == text
    assert rows[0]["job_id"] == "job-1"
    assert rows[0]["company"] == "Example Co"
    assert rows[0]["final_role_label"] == "Data Scientist / ML"
    assert rows[0]["final_ai_signal_label"] == ""


def test_apply_adjudications_rejects_missing_final_disputed_label():
    labels = pd.DataFrame(
        [
            {
                "task_hash": "abc",
                "job_id": "1",
                "role_label": "Data Scientist / ML",
                "entry_fit_label": "Unclear",
                "ai_signal_label": "AI-adjacent but vague",
                "coding_signal_label": "No coding signal",
            }
        ]
    )

    with pytest.raises(ValueError, match="missing final labels"):
        apply_adjudications(
            labels,
            [
                {
                    "task_hash": "abc",
                    "disagreement_dimensions": "AI_SIGNAL",
                    "final_ai_signal_label": "",
                }
            ],
        )


def test_apply_adjudications_updates_completed_final_labels():
    labels = pd.DataFrame(
        [
            {
                "task_hash": "abc",
                "job_id": "1",
                "role_label": "Data Scientist / ML",
                "entry_fit_label": "Unclear",
                "ai_signal_label": "AI-adjacent but vague",
                "coding_signal_label": "No coding signal",
            }
        ]
    )

    final, report = apply_adjudications(
        labels,
        [
            {
                "task_hash": "abc",
                "disagreement_dimensions": "AI_SIGNAL",
                "final_ai_signal_label": "Explicit AI / ML system work",
            }
        ],
    )

    assert final.iloc[0]["ai_signal_label"] == "Explicit AI / ML system work"
    assert report["changed_tasks"] == 1
    assert report["changes_by_dimension"] == {"ai_signal_label": 1}
