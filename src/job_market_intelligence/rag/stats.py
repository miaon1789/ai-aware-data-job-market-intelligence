"""Aggregate SQL over the analysis database: the counting tool.

Every query here is built from an allowlisted dimension name rather than from
model-supplied SQL. An agent tool that accepts a SQL string is an arbitrary
read primitive against a database of licensed private text, and no prompt
instruction reliably constrains it. Restricting the surface to named dimensions
means the worst a caller can do is ask for a count it was already allowed to
see.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

LABEL_DIMENSIONS = {
    "entry_fit": "entry_fit_label",
    "ai_signal": "ai_signal_label",
    "coding_signal": "coding_signal_label",
    "semantic_role": "role_label",
}

JOB_DIMENSIONS = {
    "role": "analysis_role",
    "city": "city",
    "seniority": "seniority",
    "source": "source",
    "company": "company",
}

SKILL_DIMENSIONS = ("skill", "skill_category", "skill_context", "role_skill")

DIMENSIONS = (*JOB_DIMENSIONS, *SKILL_DIMENSIONS, *LABEL_DIMENSIONS)


@dataclass(frozen=True)
class StatsResult:
    dimension: str
    rows: pd.DataFrame
    total_documents: int
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "total_documents": self.total_documents,
            "rows": self.rows.to_dict(orient="records"),
            "note": self.note,
        }


class JobStatsService:
    """Read-only aggregate queries over the corpus database."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        document_labels_path: str | Path | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.document_labels_path = (
            Path(document_labels_path) if document_labels_path is not None else None
        )

    def _connect(self):
        connection = duckdb.connect(str(self.database_path), read_only=True)
        if self.document_labels_path is not None and self.document_labels_path.exists():
            labels = pd.read_csv(self.document_labels_path, dtype=str, keep_default_na=False)
            connection.register("document_labels", labels)
        return connection

    def available_dimensions(self) -> list[str]:
        dimensions = [*JOB_DIMENSIONS, *SKILL_DIMENSIONS]
        if self.document_labels_path is not None and self.document_labels_path.exists():
            dimensions.extend(LABEL_DIMENSIONS)
        return sorted(dimensions)

    def query(
        self,
        dimension: str,
        *,
        filter_field: str | None = None,
        filter_value: str | None = None,
        limit: int = 25,
    ) -> StatsResult:
        """Count advertisements grouped by an allowlisted ``dimension``."""

        if dimension not in DIMENSIONS:
            raise ValueError(
                f"unknown dimension {dimension!r}; choose one of: "
                f"{', '.join(sorted(DIMENSIONS))}"
            )
        if limit < 1:
            raise ValueError("limit must be positive")

        with self._connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])
            if dimension in JOB_DIMENSIONS:
                rows = self._job_dimension(connection, dimension, filter_field, filter_value, limit)
                note = (
                    # analysis_role falls back to a TF-IDF classifier trained on
                    # the synthetic fixture wherever an advertisement carries no
                    # hand-assigned label, which is most of the corpus. Saying so
                    # is the difference between a caveat and a wrong number.
                    "Role is a model prediction for advertisements without a "
                    "hand-assigned label; use the semantic_role dimension for "
                    "adjudicated labels on the 244-advertisement subset."
                    if dimension == "role"
                    else ""
                )
            elif dimension in LABEL_DIMENSIONS:
                if self.document_labels_path is None:
                    raise ValueError(f"{dimension} requires document labels to be configured")
                rows = self._label_dimension(connection, dimension, limit)
                note = (
                    "Counts are over the hand-adjudicated label subset, not the "
                    "whole corpus."
                )
                total = int(
                    connection.execute(
                        "SELECT COUNT(DISTINCT job_id) FROM document_labels"
                    ).fetchone()[0]
                )
            else:
                rows = self._skill_dimension(
                    connection, dimension, filter_field, filter_value, limit
                )
                note = (
                    "Skill counts come from dictionary matching over advertisement "
                    "text; they measure mentions, not verified requirements."
                )

        rows = rows.copy()
        if total:
            rows["share_of_documents"] = (rows["documents"] / total).round(4)
        return StatsResult(dimension=dimension, rows=rows, total_documents=total, note=note)

    def _job_dimension(self, connection, dimension, filter_field, filter_value, limit):
        column = JOB_DIMENSIONS[dimension]
        where, parameters = "", []
        if filter_field:
            if filter_field not in JOB_DIMENSIONS:
                raise ValueError(f"cannot filter on {filter_field!r}")
            where = f" WHERE {JOB_DIMENSIONS[filter_field]} = ?"
            parameters.append(str(filter_value))
        parameters.append(int(limit))
        return connection.execute(
            f"""
            SELECT {column} AS value, COUNT(DISTINCT job_id) AS documents
            FROM jobs{where}
            GROUP BY {column}
            ORDER BY documents DESC, value
            LIMIT ?
            """,
            parameters,
        ).fetchdf()

    def _label_dimension(self, connection, dimension, limit):
        column = LABEL_DIMENSIONS[dimension]
        return connection.execute(
            f"""
            SELECT {column} AS value, COUNT(DISTINCT job_id) AS documents
            FROM document_labels
            GROUP BY {column}
            ORDER BY documents DESC, value
            LIMIT ?
            """,
            [int(limit)],
        ).fetchdf()

    def _skill_dimension(self, connection, dimension, filter_field, filter_value, limit):
        if dimension == "role_skill":
            where, parameters = "", []
            if filter_field == "skill" and filter_value:
                where = " WHERE m.skill = ?"
                parameters.append(str(filter_value))
            parameters.append(int(limit))
            return connection.execute(
                f"""
                SELECT j.analysis_role || ' / ' || m.skill AS value,
                       COUNT(DISTINCT m.job_id) AS documents
                FROM skill_mentions m JOIN jobs j USING (job_id){where}
                GROUP BY j.analysis_role, m.skill
                ORDER BY documents DESC, value
                LIMIT ?
                """,
                parameters,
            ).fetchdf()

        column = {
            "skill": "skill",
            "skill_category": "category",
            "skill_context": "context",
        }[dimension]
        conditions, parameters = [], []
        for field, value in (("skill", filter_value if filter_field == "skill" else None),
                             ("category", filter_value if filter_field == "category" else None),
                             ("context", filter_value if filter_field == "context" else None)):
            if value:
                conditions.append(f"{field} = ?")
                parameters.append(str(value))
        if filter_field and filter_field not in {"skill", "category", "context"}:
            raise ValueError(f"cannot filter skills on {filter_field!r}")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        parameters.append(int(limit))
        return connection.execute(
            f"""
            SELECT {column} AS value, COUNT(DISTINCT job_id) AS documents
            FROM skill_mentions{where}
            GROUP BY {column}
            ORDER BY documents DESC, value
            LIMIT ?
            """,
            parameters,
        ).fetchdf()
