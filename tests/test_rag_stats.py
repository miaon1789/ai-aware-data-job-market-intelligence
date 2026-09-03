import duckdb
import pandas as pd
import pytest

from job_market_intelligence.rag.stats import DIMENSIONS, JobStatsService


@pytest.fixture
def service(tmp_path):
    jobs = pd.DataFrame(
        [
            {"job_id": "a", "analysis_role": "Data Engineer", "city": "Sydney",
             "seniority": "Non-Junior", "source": "greenhouse-ats", "company": "Northwind"},
            {"job_id": "b", "analysis_role": "Data Analyst", "city": "Melbourne",
             "seniority": "Graduate / Junior", "source": "adzuna-api", "company": "Harbour"},
            {"job_id": "c", "analysis_role": "Data Engineer", "city": "Sydney",
             "seniority": "Unclear", "source": "adzuna-api", "company": "Harbour"},
        ]
    )
    mentions = pd.DataFrame(
        [
            {"job_id": "a", "skill": "SQL", "category": "Data", "context": "required"},
            {"job_id": "b", "skill": "SQL", "category": "Data", "context": "preferred"},
            {"job_id": "a", "skill": "dbt", "category": "Data Engineering",
             "context": "required"},
        ]
    )
    database = tmp_path / "corpus.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.register("jobs_frame", jobs)
        connection.register("mentions_frame", mentions)
        connection.execute("CREATE TABLE jobs AS SELECT * FROM jobs_frame")
        connection.execute("CREATE TABLE skill_mentions AS SELECT * FROM mentions_frame")

    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {"job_id": "a", "role_label": "Data Engineer", "entry_fit_label": "Experienced role",
             "ai_signal_label": "No AI signal", "coding_signal_label": "Production engineering"},
            {"job_id": "b", "role_label": "Data Analyst / BI",
             "entry_fit_label": "Likely entry-level", "ai_signal_label": "No AI signal",
             "coding_signal_label": "No coding signal"},
        ]
    ).to_csv(labels, index=False)
    return JobStatsService(database, document_labels_path=labels)


def test_counts_group_by_an_allowlisted_dimension(service):
    result = service.query("city")
    rows = {row["value"]: row["documents"] for row in result.rows.to_dict("records")}
    assert rows == {"Sydney": 2, "Melbourne": 1}
    assert result.total_documents == 3
    assert result.rows["share_of_documents"].sum() == pytest.approx(1.0)


def test_skill_counts_are_marked_as_mentions_not_requirements(service):
    result = service.query("skill")
    assert "mentions" in result.note
    rows = {row["value"]: row["documents"] for row in result.rows.to_dict("records")}
    assert rows["SQL"] == 2


def test_skill_context_distinguishes_required_from_preferred(service):
    result = service.query("skill_context", filter_field="skill", filter_value="SQL")
    rows = {row["value"]: row["documents"] for row in result.rows.to_dict("records")}
    assert rows == {"required": 1, "preferred": 1}


def test_role_dimension_warns_that_it_is_a_prediction(service):
    assert "model prediction" in service.query("role").note


def test_label_dimensions_are_scoped_to_the_labelled_subset(service):
    result = service.query("entry_fit")
    assert result.total_documents == 2
    assert "label subset" in result.note


def test_role_skill_breaks_a_skill_down_by_role(service):
    result = service.query("role_skill", filter_field="skill", filter_value="SQL")
    values = set(result.rows["value"])
    assert values == {"Data Engineer / SQL", "Data Analyst / SQL"}


def test_unknown_dimensions_are_refused_with_the_allowlist(service):
    with pytest.raises(ValueError) as excinfo:
        service.query("salary_band")
    assert "unknown dimension" in str(excinfo.value)
    assert "city" in str(excinfo.value)


def test_filters_are_restricted_to_known_fields(service):
    with pytest.raises(ValueError):
        service.query("city", filter_field="description", filter_value="x")
    with pytest.raises(ValueError):
        service.query("skill", filter_field="job_id", filter_value="a")


def test_limit_must_be_positive(service):
    with pytest.raises(ValueError):
        service.query("city", limit=0)


def test_every_advertised_dimension_actually_runs(service):
    for dimension in service.available_dimensions():
        assert dimension in DIMENSIONS
        result = service.query(dimension)
        assert set(result.rows.columns) >= {"value", "documents"}


def test_label_dimensions_are_hidden_when_labels_are_absent(tmp_path, service):
    bare = JobStatsService(service.database_path)
    assert "entry_fit" not in bare.available_dimensions()
    with pytest.raises(ValueError):
        bare.query("entry_fit")
