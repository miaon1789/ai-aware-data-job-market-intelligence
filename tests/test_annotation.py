import json

import pandas as pd

from job_market_intelligence.annotation import (
    prepare_annotation_package,
    write_annotation_package,
)
from scripts.generate_synthetic_data import generate_records


def synthetic_jobs():
    jobs = pd.DataFrame(generate_records()[:-3])
    jobs["analysis_role"] = jobs["role_label"]
    return jobs


def test_annotation_package_is_deterministic_and_company_disjoint(tmp_path):
    first = prepare_annotation_package(
        synthetic_jobs(), target_documents=30, target_skill_documents=15, seed=7
    )
    second = prepare_annotation_package(
        synthetic_jobs(), target_documents=30, target_skill_documents=15, seed=7
    )
    pd.testing.assert_frame_equal(first.manifest, second.manifest)
    assert len(first.document_tasks) == 30
    assert len(first.skill_tasks) == 15
    assert first.manifest["task_hash"].is_unique

    company_splits = first.manifest.groupby("company_group")["split"].nunique()
    assert company_splits.max() == 1
    assert set(first.manifest["split"]) == {"train", "dev", "test"}

    write_annotation_package(first, tmp_path)
    task = json.loads(
        (tmp_path / "document_classification.jsonl").read_text().splitlines()[0]
    )
    assert set(task) == {"text"}
    assert (tmp_path / "manifest.csv").exists()
    for split in ("train", "dev", "test"):
        assert (tmp_path / f"document_classification_{split}.jsonl").exists()
        assert (tmp_path / f"skill_sequence_labeling_{split}.jsonl").exists()


def test_annotation_company_cap_is_enforced():
    package = prepare_annotation_package(
        synthetic_jobs(),
        target_documents=54,
        target_skill_documents=20,
        maximum_per_company=2,
    )
    counts = package.manifest["company_group"].value_counts()
    assert counts.max() <= 2


def test_annotation_package_drops_duplicate_task_text():
    jobs = synthetic_jobs()
    duplicate = jobs.iloc[[0]].copy()
    duplicate["job_id"] = "duplicate-text"
    duplicate["company"] = "Duplicate Company"
    jobs = pd.concat([jobs, duplicate], ignore_index=True)

    package = prepare_annotation_package(
        jobs,
        target_documents=len(jobs),
        target_skill_documents=20,
    )

    assert package.manifest["task_hash"].is_unique
    assert len(package.document_tasks) == len({task["text"] for task in package.document_tasks})


def test_annotation_group_minimum_oversamples_small_collection_groups():
    jobs = synthetic_jobs()
    jobs["collection_group"] = "general_baseline"
    jobs.loc[:5, "collection_group"] = "entry_level"

    package = prepare_annotation_package(
        jobs,
        target_documents=30,
        target_skill_documents=10,
        minimum_per_collection_group=6,
    )

    counts = package.manifest["collection_group"].value_counts()
    assert counts["entry_level"] == 6
