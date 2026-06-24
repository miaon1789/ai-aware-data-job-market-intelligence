"""Validate and normalise private Doccano annotation exports."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from .annotation import task_hash

ROLE_LABELS = {
    "ROLE::Data Analyst / BI",
    "ROLE::Data Engineer",
    "ROLE::Data Scientist / ML",
    "ROLE::AI / Automation",
    "ROLE::Other / Mixed",
}
ENTRY_FIT_LABELS = {
    "ENTRY_FIT::Likely entry-level",
    "ENTRY_FIT::Experienced role",
    "ENTRY_FIT::Unclear",
}
AI_SIGNAL_LABELS = {
    "AI_SIGNAL::Explicit GenAI / LLM tool",
    "AI_SIGNAL::Explicit AI / ML system work",
    "AI_SIGNAL::Workflow automation",
    "AI_SIGNAL::AI-adjacent but vague",
    "AI_SIGNAL::No AI signal",
    "AI_SIGNAL::Unclear",
}
CODING_SIGNAL_LABELS = {
    "CODING_SIGNAL::Standalone coding skill",
    "CODING_SIGNAL::Automation / scripting",
    "CODING_SIGNAL::Production engineering",
    "CODING_SIGNAL::No coding signal",
}
SKILL_LABELS = {
    "SKILL_REQUIRED",
    "SKILL_PREFERRED",
    "SKILL_MENTIONED",
    "SKILL_NEGATED",
    "SKILL_ALTERNATIVE",
}
VALID_SPLITS = {"train", "dev", "test"}


class AnnotationValidationError(ValueError):
    """Raised when an annotation export violates the frozen label contract."""


@dataclass
class ProcessedAnnotations:
    document_labels: pd.DataFrame
    skill_labels: list[dict[str, object]]
    report: dict[str, object]


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnnotationValidationError(
                    f"{path.name}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise AnnotationValidationError(
                    f"{path.name}:{line_number}: each JSONL row must be an object"
                )
            rows.append((line_number, row))
    return rows


def _load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "task_hash",
        "job_id",
        "split",
        "selected_for_skill_spans",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise AnnotationValidationError(
            f"manifest is missing columns: {', '.join(missing)}"
        )
    if manifest.empty:
        raise AnnotationValidationError("manifest contains no tasks")
    if manifest["task_hash"].duplicated().any():
        raise AnnotationValidationError("manifest task_hash values must be unique")
    invalid_splits = sorted(set(manifest["split"]) - VALID_SPLITS)
    if invalid_splits:
        raise AnnotationValidationError(
            f"manifest contains invalid splits: {', '.join(invalid_splits)}"
        )
    truthy = {"true", "1", "yes"}
    falsy = {"false", "0", "no"}
    raw_flags = manifest["selected_for_skill_spans"].str.lower()
    invalid_flags = sorted(set(raw_flags) - truthy - falsy)
    if invalid_flags:
        raise AnnotationValidationError(
            "manifest selected_for_skill_spans contains invalid booleans"
        )
    manifest["selected_for_skill_spans"] = raw_flags.isin(truthy)
    return manifest


def _labels_from_row(row: dict[str, Any], location: str) -> list[Any]:
    labels = row.get("label", row.get("labels"))
    if not isinstance(labels, list):
        raise AnnotationValidationError(f"{location}: label/labels must be a list")
    return labels


def _annotator_id(row: dict[str, Any], source: Path) -> str:
    raw = str(
        row.get("username")
        or row.get("annotator")
        or row.get("user")
        or source.stem
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"annotator-{digest}"


def _match_task(
    row: dict[str, Any],
    manifest_by_hash: dict[str, dict[str, Any]],
    location: str,
    allowed_splits: set[str],
) -> tuple[str, dict[str, Any], str]:
    text = row.get("text")
    if not isinstance(text, str) or not text:
        raise AnnotationValidationError(f"{location}: non-empty text is required")
    digest = task_hash(text)
    task = manifest_by_hash.get(digest)
    if task is None:
        raise AnnotationValidationError(
            f"{location}: text hash is not present in the frozen manifest"
        )
    split = str(task["split"])
    if split not in allowed_splits:
        raise AnnotationValidationError(
            f"{location}: split '{split}' is locked; use an explicit test-unlock option"
        )
    return digest, task, text


def _document_records(
    paths: list[Path],
    manifest_by_hash: dict[str, dict[str, Any]],
    allowed_splits: set[str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for line_number, row in _read_jsonl(path):
            location = f"{path.name}:{line_number}"
            digest, task, _ = _match_task(
                row, manifest_by_hash, location, allowed_splits
            )
            labels = _labels_from_row(row, location)
            if not all(isinstance(label, str) for label in labels):
                raise AnnotationValidationError(
                    f"{location}: document labels must be strings"
                )
            unknown = sorted(
                set(labels)
                - ROLE_LABELS
                - ENTRY_FIT_LABELS
                - AI_SIGNAL_LABELS
                - CODING_SIGNAL_LABELS
            )
            if unknown:
                raise AnnotationValidationError(
                    f"{location}: unknown document labels: {', '.join(unknown)}"
                )
            roles = [label for label in labels if label in ROLE_LABELS]
            entry_fits = [label for label in labels if label in ENTRY_FIT_LABELS]
            ai_signals = [label for label in labels if label in AI_SIGNAL_LABELS]
            coding_signals = [label for label in labels if label in CODING_SIGNAL_LABELS]
            if (
                len(roles) != 1
                or len(entry_fits) != 1
                or len(ai_signals) != 1
                or len(coding_signals) != 1
                or len(labels) != 4
            ):
                raise AnnotationValidationError(
                    f"{location}: select exactly one ROLE, ENTRY_FIT, AI_SIGNAL "
                    "and CODING_SIGNAL label"
                )
            annotator = _annotator_id(row, path)
            key = (digest, annotator)
            if key in seen:
                raise AnnotationValidationError(
                    f"{location}: duplicate annotation for this task and annotator"
                )
            seen.add(key)
            records.append(
                {
                    "task_hash": digest,
                    "job_id": str(task["job_id"]),
                    "split": str(task["split"]),
                    "annotator_id": annotator,
                    "role_label": roles[0].removeprefix("ROLE::"),
                    "entry_fit_label": entry_fits[0].removeprefix("ENTRY_FIT::"),
                    "ai_signal_label": ai_signals[0].removeprefix("AI_SIGNAL::"),
                    "coding_signal_label": coding_signals[0].removeprefix(
                        "CODING_SIGNAL::"
                    ),
                }
            )
    return records


def _parse_span(span: Any, location: str) -> tuple[int, int, str]:
    if isinstance(span, list) and len(span) == 3:
        start, end, label = span
    elif isinstance(span, dict):
        start = span.get("start_offset", span.get("start"))
        end = span.get("end_offset", span.get("end"))
        label = span.get("label")
    else:
        raise AnnotationValidationError(
            f"{location}: each skill label must be [start, end, label]"
        )
    if isinstance(start, bool) or isinstance(end, bool):
        raise AnnotationValidationError(f"{location}: span offsets must be integers")
    if not isinstance(start, int) or not isinstance(end, int):
        raise AnnotationValidationError(f"{location}: span offsets must be integers")
    if label not in SKILL_LABELS:
        raise AnnotationValidationError(f"{location}: unknown skill label: {label}")
    return start, end, str(label)


def _skill_records(
    paths: list[Path],
    manifest_by_hash: dict[str, dict[str, Any]],
    allowed_splits: set[str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for line_number, row in _read_jsonl(path):
            location = f"{path.name}:{line_number}"
            digest, task, text = _match_task(
                row, manifest_by_hash, location, allowed_splits
            )
            if not bool(task["selected_for_skill_spans"]):
                raise AnnotationValidationError(
                    f"{location}: task was not selected for skill-span annotation"
                )
            spans = [_parse_span(item, location) for item in _labels_from_row(row, location)]
            if len(spans) != len(set(spans)):
                raise AnnotationValidationError(f"{location}: duplicate skill span")
            for start, end, _ in spans:
                if start < 0 or end <= start or end > len(text):
                    raise AnnotationValidationError(
                        f"{location}: invalid span [{start}, {end}] for text length {len(text)}"
                    )
            annotator = _annotator_id(row, path)
            key = (digest, annotator)
            if key in seen:
                raise AnnotationValidationError(
                    f"{location}: duplicate annotation for this task and annotator"
                )
            seen.add(key)
            records.append(
                {
                    "task_hash": digest,
                    "job_id": str(task["job_id"]),
                    "split": str(task["split"]),
                    "annotator_id": annotator,
                    "spans": [
                        {"start": start, "end": end, "label": label}
                        for start, end, label in spans
                    ],
                }
            )
    return records


def _cohen_kappa(first: list[str], second: list[str]) -> float | None:
    if not first:
        return None
    observed = sum(left == right for left, right in zip(first, second, strict=True)) / len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    labels = set(first_counts) | set(second_counts)
    expected = sum(
        first_counts[label] / len(first) * second_counts[label] / len(second)
        for label in labels
    )
    if expected == 1:
        return None
    return round((observed - expected) / (1 - expected), 4)


def _document_agreement(records: list[dict[str, object]]) -> dict[str, object]:
    by_task: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_task.setdefault(str(record["task_hash"]), []).append(record)
    dimensions = [
        "role_label",
        "entry_fit_label",
        "ai_signal_label",
        "coding_signal_label",
    ]
    first_values: dict[str, list[str]] = {dimension: [] for dimension in dimensions}
    second_values: dict[str, list[str]] = {dimension: [] for dimension in dimensions}
    for task_annotations in by_task.values():
        for first, second in combinations(task_annotations, 2):
            for dimension in dimensions:
                first_values[dimension].append(str(first[dimension]))
                second_values[dimension].append(str(second[dimension]))
    paired = len(first_values["role_label"])
    exact = 0
    for index in range(paired):
        if all(
            first_values[dimension][index] == second_values[dimension][index]
            for dimension in dimensions
        ):
            exact += 1
    return {
        "paired_annotations": paired,
        "role_cohen_kappa": _cohen_kappa(
            first_values["role_label"], second_values["role_label"]
        ),
        "entry_fit_cohen_kappa": _cohen_kappa(
            first_values["entry_fit_label"], second_values["entry_fit_label"]
        ),
        "ai_signal_cohen_kappa": _cohen_kappa(
            first_values["ai_signal_label"], second_values["ai_signal_label"]
        ),
        "coding_signal_cohen_kappa": _cohen_kappa(
            first_values["coding_signal_label"],
            second_values["coding_signal_label"],
        ),
        "exact_pair_agreement": round(exact / paired, 4) if paired else None,
    }


def _span_set(record: dict[str, object]) -> set[tuple[int, int, str]]:
    spans = record["spans"]
    assert isinstance(spans, list)
    return {
        (int(span["start"]), int(span["end"]), str(span["label"]))
        for span in spans
    }


def _skill_agreement(records: list[dict[str, object]]) -> dict[str, object]:
    by_task: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_task.setdefault(str(record["task_hash"]), []).append(record)
    true_positive = false_positive = false_negative = pairs = 0
    for task_annotations in by_task.values():
        for first, second in combinations(task_annotations, 2):
            left = _span_set(first)
            right = _span_set(second)
            true_positive += len(left & right)
            false_positive += len(left - right)
            false_negative += len(right - left)
            pairs += 1
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else None
    recall = true_positive / recall_denominator if recall_denominator else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "paired_annotations": pairs,
        "exact_span_precision": round(precision, 4) if precision is not None else None,
        "exact_span_recall": round(recall, 4) if recall is not None else None,
        "exact_span_f1": round(f1, 4) if f1 is not None else None,
    }


def _coverage(
    records: list[dict[str, object]], manifest: pd.DataFrame, skill_only: bool
) -> dict[str, dict[str, int | float]]:
    task_hashes = {str(record["task_hash"]) for record in records}
    denominator = manifest
    if skill_only:
        denominator = manifest[manifest["selected_for_skill_spans"]]
    result: dict[str, dict[str, int | float]] = {}
    for split in ("train", "dev", "test"):
        eligible = set(denominator.loc[denominator["split"] == split, "task_hash"])
        labelled = len(eligible & task_hashes)
        result[split] = {
            "labelled_tasks": labelled,
            "eligible_tasks": len(eligible),
            "share": round(labelled / len(eligible), 4) if eligible else 0.0,
        }
    return result


def process_annotation_exports(
    manifest_path: Path,
    *,
    document_exports: list[Path] | None = None,
    skill_exports: list[Path] | None = None,
    include_test: bool = False,
) -> ProcessedAnnotations:
    """Validate exports and return text-free private records plus aggregate metrics."""

    document_exports = document_exports or []
    skill_exports = skill_exports or []
    if not document_exports and not skill_exports:
        raise AnnotationValidationError("provide at least one annotation export")
    manifest = _load_manifest(manifest_path)
    manifest_by_hash = {
        str(row["task_hash"]): row for row in manifest.to_dict(orient="records")
    }
    allowed_splits = VALID_SPLITS if include_test else {"train", "dev"}
    document_records = _document_records(
        document_exports, manifest_by_hash, allowed_splits
    )
    skill_records = _skill_records(skill_exports, manifest_by_hash, allowed_splits)
    document_frame = pd.DataFrame.from_records(
        document_records,
        columns=[
            "task_hash",
            "job_id",
            "split",
            "annotator_id",
            "role_label",
            "entry_fit_label",
            "ai_signal_label",
            "coding_signal_label",
        ],
    )
    role_counts = Counter(str(row["role_label"]) for row in document_records)
    entry_fit_counts = Counter(str(row["entry_fit_label"]) for row in document_records)
    ai_signal_counts = Counter(str(row["ai_signal_label"]) for row in document_records)
    coding_signal_counts = Counter(
        str(row["coding_signal_label"]) for row in document_records
    )
    skill_counts = Counter(
        str(span["label"])
        for record in skill_records
        for span in record["spans"]  # type: ignore[union-attr]
    )
    report: dict[str, object] = {
        "contract": "frozen manifest hash match and strict label validation",
        "contains_job_text": False,
        "test_split_unlocked": include_test,
        "document": {
            "annotation_rows": len(document_records),
            "unique_tasks": len({str(row["task_hash"]) for row in document_records}),
            "coverage": _coverage(document_records, manifest, skill_only=False),
            "role_distribution": dict(sorted(role_counts.items())),
            "entry_fit_distribution": dict(sorted(entry_fit_counts.items())),
            "ai_signal_distribution": dict(sorted(ai_signal_counts.items())),
            "coding_signal_distribution": dict(sorted(coding_signal_counts.items())),
            "agreement": _document_agreement(document_records),
        },
        "skill": {
            "annotation_rows": len(skill_records),
            "unique_tasks": len({str(row["task_hash"]) for row in skill_records}),
            "span_count": sum(len(row["spans"]) for row in skill_records),  # type: ignore[arg-type]
            "coverage": _coverage(skill_records, manifest, skill_only=True),
            "label_distribution": dict(sorted(skill_counts.items())),
            "agreement": _skill_agreement(skill_records),
        },
        "publication_policy": (
            "aggregate report may be published; normalized labels and source exports stay private"
        ),
    }
    return ProcessedAnnotations(document_frame, skill_records, report)


def write_processed_annotations(
    result: ProcessedAnnotations, output_directory: Path, report_path: Path
) -> None:
    """Write normalized labels privately and the text-free aggregate report separately."""

    output_directory.mkdir(parents=True, exist_ok=True)
    output_directory.chmod(0o700)
    document_path = output_directory / "document_labels.csv"
    result.document_labels.to_csv(document_path, index=False)
    document_path.chmod(0o600)
    skill_path = output_directory / "skill_labels.jsonl"
    content = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in result.skill_labels
    )
    skill_path.write_text(content, encoding="utf-8")
    skill_path.chmod(0o600)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
