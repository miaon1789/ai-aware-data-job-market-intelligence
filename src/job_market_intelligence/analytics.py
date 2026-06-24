"""DuckDB persistence and reusable analytical views."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def build_database(
    database_path: str | Path,
    jobs: pd.DataFrame,
    skill_mentions: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
) -> None:
    """Replace pipeline-owned tables and views in a DuckDB database."""

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(path)) as connection:
        connection.register("jobs_frame", jobs)
        connection.register("mentions_frame", skill_mentions)
        connection.register("duplicate_frame", duplicate_audit)
        connection.execute("CREATE OR REPLACE TABLE jobs AS SELECT * FROM jobs_frame")
        connection.execute(
            "CREATE OR REPLACE TABLE skill_mentions AS SELECT * FROM mentions_frame"
        )
        connection.execute(
            "CREATE OR REPLACE TABLE duplicate_audit AS SELECT * FROM duplicate_frame"
        )
        connection.execute(
            """
            CREATE OR REPLACE VIEW role_city_summary AS
            SELECT city, analysis_role AS role, seniority, COUNT(*) AS job_count
            FROM jobs
            GROUP BY city, analysis_role, seniority
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE VIEW role_skill_summary AS
            WITH denominators AS (
                SELECT analysis_role, city, COUNT(DISTINCT job_id) AS total_ads
                FROM jobs
                GROUP BY analysis_role, city
            )
            SELECT
                j.analysis_role AS role,
                j.city,
                m.skill,
                m.category,
                m.context,
                COUNT(DISTINCT m.job_id) AS ads_with_skill,
                COUNT(DISTINCT m.job_id) * 1.0 / d.total_ads AS share_of_ads
            FROM jobs j
            JOIN skill_mentions m USING (job_id)
            JOIN denominators d
              ON j.analysis_role = d.analysis_role AND j.city = d.city
            GROUP BY
                j.analysis_role, j.city, m.skill, m.category, m.context, d.total_ads
            """
        )


def query_frame(database_path: str | Path, query: str, parameters=None) -> pd.DataFrame:
    with duckdb.connect(str(database_path), read_only=True) as connection:
        return connection.execute(query, parameters or []).fetchdf()
