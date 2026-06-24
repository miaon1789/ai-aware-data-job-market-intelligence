"""Prepare and apply human adjudication over LLM-audit disagreements."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from .annotation import task_hash
from .llm_annotation import (
    AI_SIGNAL_VALUES,
    CODING_SIGNAL_VALUES,
    ENTRY_FIT_VALUES,
    ROLE_VALUES,
)

DIMENSIONS = {
    "ROLE": ("role_label", ROLE_VALUES),
    "ENTRY_FIT": ("entry_fit_label", ENTRY_FIT_VALUES),
    "AI_SIGNAL": ("ai_signal_label", AI_SIGNAL_VALUES),
    "CODING_SIGNAL": ("coding_signal_label", CODING_SIGNAL_VALUES),
}


def labels_to_dimensions(labels: list[str] | str) -> dict[str, str]:
    """Parse prefixed document labels into normalized dimension values."""

    if isinstance(labels, str):
        labels = [label.strip() for label in labels.split("|") if label.strip()]
    result: dict[str, str] = {}
    for label in labels:
        if "::" not in label:
            continue
        dimension, value = label.split("::", 1)
        if dimension in DIMENSIONS:
            result[DIMENSIONS[dimension][0]] = value
    return result


def split_disagreement_dimensions(value: str) -> list[str]:
    """Parse a pipe-separated disagreement dimension string."""

    return [part.strip() for part in value.split("|") if part.strip()]


def build_adjudication_template(
    *,
    human_export_rows: list[dict[str, Any]],
    disagreement_rows: list[dict[str, str]],
    source_job_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Create private adjudication rows with text and blank final conflict labels."""

    text_by_hash = {}
    for row in human_export_rows:
        text = row.get("text")
        if isinstance(text, str) and text:
            text_by_hash[task_hash(text)] = text

    source_by_job_id = {
        str(row.get("job_id")): row for row in source_job_rows or [] if row.get("job_id")
    }

    template_rows: list[dict[str, str]] = []
    for row in disagreement_rows:
        digest = row["task_hash"]
        text = text_by_hash.get(digest, "")
        human_labels = labels_to_dimensions(row["current_labels"])
        llm_labels = labels_to_dimensions(row["llm_observed_labels"])
        disagreements = set(split_disagreement_dimensions(row["disagreement_dimensions"]))
        source_row = source_by_job_id.get(str(row.get("job_id", "")), {})

        output = {
            "task_hash": digest,
            "job_id": str(row.get("job_id", "")),
            "title": row.get("title", ""),
            "source_title": str(source_row.get("title", "")),
            "company": str(source_row.get("company", "")),
            "city": str(source_row.get("city", "")),
            "posted_at": str(source_row.get("posted_at", "")),
            "collection_query": str(source_row.get("collection_query", "")),
            "collection_group": str(source_row.get("collection_group", "")),
            "source_url": str(source_row.get("source_url", "")),
            "llm_confidence": row.get("confidence", ""),
            "disagreement_dimensions": row.get("disagreement_dimensions", ""),
            "coding_likely_by_role": row.get("coding_likely_by_role", ""),
            "text": text,
        }
        for dimension, (column, _) in DIMENSIONS.items():
            output[f"human_{column}"] = human_labels.get(column, "")
            output[f"llm_{column}"] = llm_labels.get(column, "")
            output[f"final_{column}"] = (
                "" if dimension in disagreements else human_labels.get(column, "")
            )
        output["llm_rationale"] = row.get("rationale", "")
        output["adjudication_notes"] = ""
        template_rows.append(output)
    return template_rows


def _validate_final_label(value: str, column: str) -> None:
    allowed = next(values for label_column, values in DIMENSIONS.values() if label_column == column)
    if value not in allowed:
        raise ValueError(f"invalid {column}: {value}")


def apply_adjudications(
    document_labels: pd.DataFrame, adjudication_rows: list[dict[str, str]]
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply completed adjudication rows to normalized private document labels."""

    required = {
        "task_hash",
        "role_label",
        "entry_fit_label",
        "ai_signal_label",
        "coding_signal_label",
    }
    missing = sorted(required - set(document_labels.columns))
    if missing:
        raise ValueError(f"document labels are missing columns: {', '.join(missing)}")
    if document_labels["task_hash"].duplicated().any():
        raise ValueError("document labels must have unique task_hash values")

    final = document_labels.copy()
    final = final.set_index("task_hash", drop=False)
    changes_by_dimension: Counter[str] = Counter()
    changed_tasks: set[str] = set()
    missing_required: list[str] = []

    for row in adjudication_rows:
        digest = row["task_hash"]
        if digest not in final.index:
            raise ValueError(f"adjudication task_hash not found in document labels: {digest}")
        disagreements = set(split_disagreement_dimensions(row.get("disagreement_dimensions", "")))
        for dimension, (column, _) in DIMENSIONS.items():
            value = row.get(f"final_{column}", "").strip()
            if not value:
                if dimension in disagreements:
                    missing_required.append(f"{digest}:{column}")
                continue
            _validate_final_label(value, column)
            old_value = str(final.at[digest, column])
            if old_value != value:
                final.at[digest, column] = value
                changes_by_dimension[column] += 1
                changed_tasks.add(digest)

    if missing_required:
        preview = ", ".join(missing_required[:10])
        suffix = "..." if len(missing_required) > 10 else ""
        raise ValueError(f"missing final labels for disputed dimensions: {preview}{suffix}")

    output = final.reset_index(drop=True)
    report = {
        "contains_job_text": False,
        "input_label_rows": int(len(document_labels)),
        "adjudication_rows": int(len(adjudication_rows)),
        "changed_tasks": int(len(changed_tasks)),
        "changes_by_dimension": dict(sorted(changes_by_dimension.items())),
        "final_role_distribution": dict(sorted(Counter(output["role_label"]).items())),
        "final_entry_fit_distribution": dict(
            sorted(Counter(output["entry_fit_label"]).items())
        ),
        "final_ai_signal_distribution": dict(
            sorted(Counter(output["ai_signal_label"]).items())
        ),
        "final_coding_signal_distribution": dict(
            sorted(Counter(output["coding_signal_label"]).items())
        ),
    }
    return output, report
