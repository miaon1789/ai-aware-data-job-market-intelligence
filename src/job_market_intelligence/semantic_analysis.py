"""Aggregate analysis over semantic document labels and skill mentions."""

from __future__ import annotations

import pandas as pd

LABEL_DIMENSIONS = {
    "role_label": "role",
    "entry_fit_label": "entry_fit",
    "ai_signal_label": "ai_signal",
    "coding_signal_label": "coding_signal",
}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def suppress_small_cells(
    frame: pd.DataFrame, *, count_column: str = "job_count", minimum_cell_size: int = 10
) -> pd.DataFrame:
    """Return aggregate rows with counts meeting the publication threshold."""

    if minimum_cell_size < 1:
        raise ValueError("minimum_cell_size must be positive")
    if count_column not in frame.columns:
        raise ValueError(f"count column is missing: {count_column}")
    return frame[frame[count_column] >= minimum_cell_size].reset_index(drop=True)


def label_distribution(document_labels: pd.DataFrame) -> pd.DataFrame:
    """Build long-form label distributions for each document-label dimension."""

    _require_columns(
        document_labels,
        {"job_id", *LABEL_DIMENSIONS.keys()},
        "document_labels",
    )
    total = max(1, document_labels["job_id"].nunique())
    rows = []
    for column, dimension in LABEL_DIMENSIONS.items():
        counts = document_labels.groupby(column)["job_id"].nunique().reset_index()
        for row in counts.itertuples(index=False):
            rows.append(
                {
                    "dimension": dimension,
                    "label": str(getattr(row, column)),
                    "job_count": int(row.job_id),
                    "share_of_labelled_jobs": round(row.job_id / total, 4),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["dimension", "job_count", "label"], ascending=[True, False, True]
    )


def label_crosstab(
    document_labels: pd.DataFrame, row_dimension: str, column_dimension: str
) -> pd.DataFrame:
    """Build a long-form crosstab over two semantic label dimensions."""

    _require_columns(
        document_labels,
        {"job_id", row_dimension, column_dimension},
        "document_labels",
    )
    denominators = (
        document_labels.groupby(row_dimension)["job_id"]
        .nunique()
        .rename("row_total")
        .reset_index()
    )
    table = (
        document_labels.groupby([row_dimension, column_dimension])["job_id"]
        .nunique()
        .rename("job_count")
        .reset_index()
        .merge(denominators, on=row_dimension, how="left")
    )
    table["share_within_row"] = (table["job_count"] / table["row_total"]).round(4)
    return table.sort_values(
        [row_dimension, "job_count", column_dimension],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def top_skills_by_role(
    document_labels: pd.DataFrame,
    skill_mentions: pd.DataFrame,
    *,
    max_skills_per_role: int = 10,
) -> pd.DataFrame:
    """Count extracted skill mentions by semantic role label."""

    _require_columns(document_labels, {"job_id", "role_label"}, "document_labels")
    _require_columns(skill_mentions, {"job_id", "skill", "category", "context"}, "skill_mentions")
    if max_skills_per_role < 1:
        raise ValueError("max_skills_per_role must be positive")

    role_totals = (
        document_labels.groupby("role_label")["job_id"]
        .nunique()
        .rename("role_job_count")
        .reset_index()
    )
    joined = skill_mentions.merge(
        document_labels[["job_id", "role_label"]].drop_duplicates(),
        on="job_id",
        how="inner",
    )
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "role_label",
                "skill",
                "category",
                "top_context",
                "job_count",
                "role_job_count",
                "share_within_role",
            ]
        )
    counts = (
        joined.groupby(["role_label", "skill", "category"])["job_id"]
        .nunique()
        .rename("job_count")
        .reset_index()
        .merge(role_totals, on="role_label", how="left")
    )
    top_context = (
        joined.groupby(["role_label", "skill", "category"])["context"]
        .agg(lambda values: values.value_counts().index[0])
        .rename("top_context")
        .reset_index()
    )
    counts = counts.merge(top_context, on=["role_label", "skill", "category"], how="left")
    counts["share_within_role"] = (
        counts["job_count"] / counts["role_job_count"]
    ).round(4)
    counts = counts.sort_values(
        ["role_label", "job_count", "skill"],
        ascending=[True, False, True],
    )
    return (
        counts.groupby("role_label", group_keys=False)
        .head(max_skills_per_role)
        .reset_index(drop=True)
    )


def build_semantic_tables(
    document_labels: pd.DataFrame,
    skill_mentions: pd.DataFrame,
    *,
    max_skills_per_role: int = 10,
) -> dict[str, pd.DataFrame]:
    """Build all aggregate semantic-analysis tables."""

    return {
        "label_distribution": label_distribution(document_labels),
        "role_ai_signal": label_crosstab(
            document_labels, "role_label", "ai_signal_label"
        ),
        "entry_ai_signal": label_crosstab(
            document_labels, "entry_fit_label", "ai_signal_label"
        ),
        "role_coding_signal": label_crosstab(
            document_labels, "role_label", "coding_signal_label"
        ),
        "top_skills_by_role": top_skills_by_role(
            document_labels,
            skill_mentions,
            max_skills_per_role=max_skills_per_role,
        ),
    }
