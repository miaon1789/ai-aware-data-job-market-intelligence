import json

import pandas as pd
import pytest

from job_market_intelligence.annotation import task_hash
from job_market_intelligence.annotation_results import (
    AnnotationValidationError,
    process_annotation_exports,
    write_processed_annotations,
)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def annotation_fixture(tmp_path):
    texts = {
        "train": "Data Analyst\n\nSQL and Power BI are required for this role.",
        "dev": "Data Engineer\n\nBuild reliable Python data pipelines.",
        "test": "Data Scientist\n\nDevelop statistical models in Python.",
    }
    manifest = pd.DataFrame(
        [
            {
                "task_hash": task_hash(text),
                "job_id": f"job-{split}",
                "split": split,
                "selected_for_skill_spans": True,
            }
            for split, text in texts.items()
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return texts, manifest_path


def document_row(
    text,
    username,
    role="Data Analyst / BI",
    entry_fit="Likely entry-level",
    ai_signal="No AI signal",
    coding_signal="Standalone coding skill",
):
    return {
        "text": text,
        "label": [
            f"ROLE::{role}",
            f"ENTRY_FIT::{entry_fit}",
            f"AI_SIGNAL::{ai_signal}",
            f"CODING_SIGNAL::{coding_signal}",
        ],
        "username": username,
    }


def test_processes_exports_and_reports_agreement_without_text(
    tmp_path, annotation_fixture
):
    texts, manifest_path = annotation_fixture
    first = write_jsonl(
        tmp_path / "documents-first.jsonl",
        [
            document_row(texts["train"], "alice"),
            document_row(texts["dev"], "alice", role="Data Engineer"),
        ],
    )
    second = write_jsonl(
        tmp_path / "documents-second.jsonl",
        [
            document_row(texts["train"], "bob"),
            document_row(
                texts["dev"],
                "bob",
                role="Data Engineer",
                entry_fit="Experienced role",
            ),
        ],
    )
    sql_start = texts["train"].index("SQL")
    skills_first = write_jsonl(
        tmp_path / "skills-first.jsonl",
        [
            {
                "text": texts["train"],
                "label": [[sql_start, sql_start + 3, "SKILL_REQUIRED"]],
                "username": "alice",
            }
        ],
    )
    skills_second = write_jsonl(
        tmp_path / "skills-second.jsonl",
        [
            {
                "text": texts["train"],
                "labels": [[sql_start, sql_start + 3, "SKILL_REQUIRED"]],
                "username": "bob",
            }
        ],
    )

    result = process_annotation_exports(
        manifest_path,
        document_exports=[first, second],
        skill_exports=[skills_first, skills_second],
    )

    assert len(result.document_labels) == 4
    assert result.report["contains_job_text"] is False
    assert result.report["document"]["agreement"]["paired_annotations"] == 2
    assert result.report["document"]["agreement"]["exact_pair_agreement"] == 0.5
    assert result.report["skill"]["agreement"]["exact_span_f1"] == 1.0
    assert result.report["document"]["coverage"]["test"]["labelled_tasks"] == 0
    serialized = json.dumps(result.report)
    assert "Build reliable Python" not in serialized
    assert "alice" not in serialized

    output = tmp_path / "processed"
    report = tmp_path / "annotation_quality.json"
    write_processed_annotations(result, output, report)
    assert "text" not in pd.read_csv(output / "document_labels.csv").columns
    assert "Data Analyst\n\n" not in (output / "skill_labels.jsonl").read_text()


def test_rejects_test_annotations_until_explicitly_unlocked(
    tmp_path, annotation_fixture
):
    texts, manifest_path = annotation_fixture
    export = write_jsonl(
        tmp_path / "test-export.jsonl",
        [document_row(texts["test"], "alice", role="Data Scientist / ML")],
    )
    with pytest.raises(AnnotationValidationError, match="split 'test' is locked"):
        process_annotation_exports(manifest_path, document_exports=[export])

    result = process_annotation_exports(
        manifest_path, document_exports=[export], include_test=True
    )
    assert result.report["test_split_unlocked"] is True


def test_rejects_incomplete_document_labels(tmp_path, annotation_fixture):
    texts, manifest_path = annotation_fixture
    export = write_jsonl(
        tmp_path / "invalid-document.jsonl",
        [{"text": texts["train"], "label": ["ROLE::Data Analyst / BI"]}],
    )
    with pytest.raises(AnnotationValidationError, match="exactly one ROLE"):
        process_annotation_exports(manifest_path, document_exports=[export])


def test_rejects_invalid_skill_offsets(tmp_path, annotation_fixture):
    texts, manifest_path = annotation_fixture
    export = write_jsonl(
        tmp_path / "invalid-skills.jsonl",
        [
            {
                "text": texts["train"],
                "label": [[0, len(texts["train"]) + 1, "SKILL_REQUIRED"]],
            }
        ],
    )
    with pytest.raises(AnnotationValidationError, match="invalid span"):
        process_annotation_exports(manifest_path, skill_exports=[export])


def test_accepts_all_document_role_labels(tmp_path, annotation_fixture):
    texts, manifest_path = annotation_fixture
    export = write_jsonl(
        tmp_path / "mixed-role.jsonl",
        [document_row(texts["train"], "alice", role="Other / Mixed")],
    )
    result = process_annotation_exports(manifest_path, document_exports=[export])
    assert result.document_labels.loc[0, "role_label"] == "Other / Mixed"
