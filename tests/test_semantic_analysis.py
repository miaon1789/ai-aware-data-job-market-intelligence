import pandas as pd
import pytest

from job_market_intelligence.semantic_analysis import (
    build_semantic_tables,
    label_crosstab,
    label_distribution,
    suppress_small_cells,
    top_skills_by_role,
)


def sample_labels():
    return pd.DataFrame(
        [
            {
                "job_id": "1",
                "role_label": "Data Analyst / BI",
                "entry_fit_label": "Likely entry-level",
                "ai_signal_label": "No AI signal",
                "coding_signal_label": "No coding signal",
            },
            {
                "job_id": "2",
                "role_label": "Data Analyst / BI",
                "entry_fit_label": "Unclear",
                "ai_signal_label": "No AI signal",
                "coding_signal_label": "Standalone coding skill",
            },
            {
                "job_id": "3",
                "role_label": "AI / Automation",
                "entry_fit_label": "Experienced role",
                "ai_signal_label": "Explicit GenAI / LLM tool",
                "coding_signal_label": "Automation / scripting",
            },
        ]
    )


def test_label_distribution_is_long_form():
    distribution = label_distribution(sample_labels())

    role_rows = distribution[distribution["dimension"] == "role"]
    analyst = role_rows[role_rows["label"] == "Data Analyst / BI"].iloc[0]
    assert analyst["job_count"] == 2
    assert analyst["share_of_labelled_jobs"] == 0.6667


def test_label_crosstab_counts_and_shares():
    table = label_crosstab(sample_labels(), "role_label", "ai_signal_label")

    analyst_no_ai = table[
        (table["role_label"] == "Data Analyst / BI")
        & (table["ai_signal_label"] == "No AI signal")
    ].iloc[0]
    assert analyst_no_ai["job_count"] == 2
    assert analyst_no_ai["share_within_row"] == 1.0


def test_top_skills_by_role():
    mentions = pd.DataFrame(
        [
            {"job_id": "1", "skill": "SQL", "category": "technical", "context": "required"},
            {"job_id": "2", "skill": "SQL", "category": "technical", "context": "mentioned"},
            {"job_id": "3", "skill": "RAG", "category": "ai", "context": "required"},
        ]
    )

    top = top_skills_by_role(sample_labels(), mentions)

    sql = top[(top["role_label"] == "Data Analyst / BI") & (top["skill"] == "SQL")]
    assert not sql.empty
    assert sql.iloc[0]["job_count"] == 2
    assert sql.iloc[0]["share_within_role"] == 1.0


def test_suppresses_small_cells():
    frame = pd.DataFrame({"label": ["a", "b"], "job_count": [9, 10]})

    result = suppress_small_cells(frame, minimum_cell_size=10)

    assert result["label"].tolist() == ["b"]


def test_build_semantic_tables_returns_expected_names():
    mentions = pd.DataFrame(
        [
            {"job_id": "1", "skill": "SQL", "category": "technical", "context": "required"},
            {"job_id": "3", "skill": "RAG", "category": "ai", "context": "required"},
        ]
    )

    tables = build_semantic_tables(sample_labels(), mentions)

    assert set(tables) == {
        "label_distribution",
        "role_ai_signal",
        "entry_ai_signal",
        "role_coding_signal",
        "top_skills_by_role",
    }


def test_rejects_missing_columns():
    with pytest.raises(ValueError, match="document_labels is missing"):
        label_distribution(pd.DataFrame({"job_id": ["1"]}))
