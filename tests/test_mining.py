import pandas as pd

from job_market_intelligence.mining import (
    cluster_job_texts,
    mine_skill_association_rules,
)


def test_clusters_jobs_without_exposing_text_columns():
    jobs = pd.DataFrame(
        [
            {
                "job_id": "a",
                "title": "Data Analyst",
                "description": "Build dashboards with SQL and Power BI.",
                "city": "Sydney",
                "analysis_role": "Data Analyst / BI",
                "seniority": "entry",
            },
            {
                "job_id": "b",
                "title": "BI Analyst",
                "description": "Create reporting and stakeholder insights.",
                "city": "Sydney",
                "analysis_role": "Data Analyst / BI",
                "seniority": "entry",
            },
            {
                "job_id": "c",
                "title": "Machine Learning Engineer",
                "description": "Deploy machine learning models and MLOps pipelines.",
                "city": "Melbourne",
                "analysis_role": "Data Scientist / ML",
                "seniority": "experienced",
            },
        ]
    )

    result = cluster_job_texts(jobs, n_clusters=2, terms_per_cluster=4, random_state=7)

    assert set(result.assignments.columns) == {
        "job_id",
        "cluster_id",
        "cluster_distance",
    }
    assert len(result.assignments) == 3
    assert "title" not in result.summary.columns
    assert "description" not in result.summary.columns
    assert result.summary["job_count"].sum() == 3


def test_mines_pairwise_skill_association_rules():
    mentions = pd.DataFrame(
        [
            {"job_id": "a", "skill": "SQL"},
            {"job_id": "a", "skill": "Power BI"},
            {"job_id": "b", "skill": "SQL"},
            {"job_id": "b", "skill": "Power BI"},
            {"job_id": "c", "skill": "Python"},
            {"job_id": "c", "skill": "Machine Learning"},
        ]
    )

    rules = mine_skill_association_rules(
        mentions,
        total_jobs=3,
        min_support=0.3,
        min_confidence=0.5,
    )

    sql_to_bi = rules[
        (rules["antecedent"] == "SQL") & (rules["consequent"] == "Power BI")
    ]
    assert not sql_to_bi.empty
    assert sql_to_bi.iloc[0]["support"] == 0.6667
    assert sql_to_bi.iloc[0]["confidence"] == 1.0
    assert sql_to_bi.iloc[0]["lift"] > 1.0
