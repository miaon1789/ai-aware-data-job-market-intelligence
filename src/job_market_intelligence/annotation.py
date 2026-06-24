"""Prepare private, leakage-aware human annotation packages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


@dataclass
class AnnotationPackage:
    manifest: pd.DataFrame
    document_tasks: list[dict[str, str]]
    skill_tasks: list[dict[str, str]]
    summary: dict[str, object]


def _task_text(row: pd.Series) -> str:
    return f"{row['title'].strip()}\n\n{row['description'].strip()}"


def task_hash(text: str) -> str:
    """Return the stable identifier used to reconnect private Doccano exports."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _cap_companies(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    parts = []
    for offset, (_, group) in enumerate(frame.groupby("company", sort=True)):
        sample_size = min(maximum, len(group))
        parts.append(group.sample(n=sample_size, random_state=seed + offset))
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()


def _stratified_sample(
    frame: pd.DataFrame, target: int, strata: pd.Series, seed: int
) -> pd.DataFrame:
    if len(frame) <= target:
        return frame.copy()
    counts = strata.value_counts()
    if (
        len(counts) > 1
        and counts.min() >= 2
        and target >= len(counts)
        and len(frame) - target >= len(counts)
    ):
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=target, random_state=seed)
        selected, _ = next(splitter.split(frame, strata))
        return frame.iloc[selected].copy()
    return frame.sample(n=target, random_state=seed).copy()


def _sample_with_group_minimum(
    frame: pd.DataFrame,
    target: int,
    strata: pd.Series,
    group_column: str,
    minimum_per_group: int,
    seed: int,
) -> pd.DataFrame:
    if minimum_per_group <= 0:
        return _stratified_sample(frame, target, strata, seed)
    if len(frame) <= target:
        return frame.copy()

    selected_parts = []
    for offset, (_, group) in enumerate(frame.groupby(group_column, sort=True)):
        sample_size = min(minimum_per_group, len(group))
        group_strata = strata.loc[group.index]
        selected_parts.append(
            _stratified_sample(group, sample_size, group_strata, seed + offset)
        )
    selected = (
        pd.concat(selected_parts).drop_duplicates("job_id")
        if selected_parts
        else frame.iloc[0:0].copy()
    )
    if len(selected) >= target:
        return _stratified_sample(selected, target, strata.loc[selected.index], seed)

    remaining = frame.drop(index=selected.index)
    fill_target = target - len(selected)
    fill = _stratified_sample(
        remaining, fill_target, strata.loc[remaining.index], seed + 10_000
    )
    return pd.concat([selected, fill]).copy()


def _distribution_score(frame: pd.DataFrame, indexes, target_share: float) -> float:
    selected = frame.iloc[indexes]
    overall = frame["analysis_role"].value_counts(normalize=True)
    observed = selected["analysis_role"].value_counts(normalize=True).reindex(
        overall.index, fill_value=0
    )
    distribution_error = float((overall - observed).abs().sum())
    size_error = abs(len(selected) / len(frame) - target_share)
    return distribution_error + size_error


def _best_group_holdout(
    frame: pd.DataFrame, *, holdout_share: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame["company"].nunique() < 2:
        return frame.copy(), frame.iloc[0:0].copy()
    best = None
    best_score = float("inf")
    for attempt in range(100):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=holdout_share, random_state=seed + attempt
        )
        retained, held_out = next(
            splitter.split(frame, groups=frame["company"])
        )
        score = _distribution_score(frame, held_out, holdout_share)
        if score < best_score:
            best = (retained, held_out)
            best_score = score
    assert best is not None
    retained, held_out = best
    return frame.iloc[retained].copy(), frame.iloc[held_out].copy()


def prepare_annotation_package(
    jobs: pd.DataFrame,
    *,
    target_documents: int = 300,
    target_skill_documents: int = 150,
    maximum_per_company: int = 3,
    minimum_per_collection_group: int = 0,
    seed: int = 42,
) -> AnnotationPackage:
    """Sample documents and create company-disjoint train/dev/test splits."""

    required = {"job_id", "title", "description", "city", "company", "analysis_role"}
    missing = sorted(required - set(jobs.columns))
    if missing:
        raise ValueError(f"jobs are missing annotation columns: {', '.join(missing)}")
    if target_documents < 1 or target_skill_documents < 1 or maximum_per_company < 1:
        raise ValueError("annotation sample sizes and company cap must be positive")
    if minimum_per_collection_group < 0:
        raise ValueError("minimum_per_collection_group cannot be negative")

    candidates = jobs.drop_duplicates("job_id").copy()
    if "collection_group" not in candidates:
        candidates["collection_group"] = "unspecified"
    candidates["collection_group"] = (
        candidates["collection_group"].astype(str).replace("", "unspecified")
    )
    candidates = candidates[
        candidates["title"].astype(str).str.len().ge(2)
        & candidates["description"].astype(str).str.len().ge(20)
    ]
    candidates["text"] = candidates.apply(_task_text, axis=1)
    candidates["task_hash"] = candidates["text"].map(task_hash)
    candidates = candidates.drop_duplicates("task_hash").copy()
    candidates = _cap_companies(candidates, maximum_per_company, seed)
    strata = (
        candidates["collection_group"].astype(str)
        + " | "
        + candidates["city"].astype(str)
        + " | "
        + candidates["analysis_role"].astype(str)
    )
    sampled = _sample_with_group_minimum(
        candidates,
        target_documents,
        strata,
        "collection_group",
        minimum_per_collection_group,
        seed,
    )

    train_dev, test = _best_group_holdout(sampled, holdout_share=0.2, seed=seed)
    # 12.5% of the remaining 80% produces an overall dev target near 10%.
    train, dev = _best_group_holdout(train_dev, holdout_share=0.125, seed=seed + 10_000)
    train["split"] = "train"
    dev["split"] = "dev"
    test["split"] = "test"
    sampled = pd.concat([train, dev, test], ignore_index=True)

    company_codes = {
        company: f"company-{index:04d}"
        for index, company in enumerate(sorted(sampled["company"].unique()), start=1)
    }
    sampled["company_group"] = sampled["company"].map(company_codes)

    skill_strata = (
        sampled["split"].astype(str)
        + " | "
        + sampled["collection_group"].astype(str)
        + " | "
        + sampled["city"].astype(str)
        + " | "
        + sampled["analysis_role"].astype(str)
    )
    skill_sample = _stratified_sample(
        sampled,
        min(target_skill_documents, len(sampled)),
        skill_strata,
        seed + 20_000,
    )
    skill_hashes = set(skill_sample["task_hash"])
    sampled["selected_for_skill_spans"] = sampled["task_hash"].isin(skill_hashes)

    manifest_columns = [
        "task_hash",
        "job_id",
        "split",
        "city",
        "analysis_role",
        "collection_group",
        "company_group",
        "selected_for_skill_spans",
    ]
    manifest = sampled[manifest_columns].sort_values(["split", "task_hash"])
    document_tasks = [
        {"text": text}
        for text in sampled.sort_values("task_hash")["text"].tolist()
    ]
    skill_tasks = [
        {"text": text}
        for text in skill_sample.sort_values("task_hash")["text"].tolist()
    ]
    split_counts = manifest["split"].value_counts().to_dict()
    role_counts = manifest["analysis_role"].value_counts().to_dict()
    group_counts = manifest["collection_group"].value_counts().to_dict()
    summary: dict[str, object] = {
        "seed": seed,
        "document_tasks": len(document_tasks),
        "skill_span_tasks": len(skill_tasks),
        "maximum_per_company": maximum_per_company,
        "minimum_per_collection_group": minimum_per_collection_group,
        "companies": int(manifest["company_group"].nunique()),
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "role_counts": {key: int(value) for key, value in role_counts.items()},
        "collection_group_counts": {
            key: int(value) for key, value in group_counts.items()
        },
        "company_disjoint_splits": True,
        "contains_private_job_text": True,
        "publication_policy": "do not publish tasks, manifest, or annotation exports",
    }
    return AnnotationPackage(
        manifest=manifest.reset_index(drop=True),
        document_tasks=document_tasks,
        skill_tasks=skill_tasks,
        summary=summary,
    )


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def write_annotation_package(package: AnnotationPackage, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    output_directory.chmod(0o700)
    package.manifest.to_csv(output_directory / "manifest.csv", index=False)
    (output_directory / "manifest.csv").chmod(0o600)
    _write_jsonl(output_directory / "document_classification.jsonl", package.document_tasks)
    _write_jsonl(output_directory / "skill_sequence_labeling.jsonl", package.skill_tasks)
    split_by_hash = package.manifest.set_index("task_hash")["split"].to_dict()
    for split in ("train", "dev", "test"):
        document_split = [
            task
            for task in package.document_tasks
            if split_by_hash[task_hash(task["text"])] == split
        ]
        skill_split = [
            task for task in package.skill_tasks if split_by_hash[task_hash(task["text"])] == split
        ]
        _write_jsonl(
            output_directory / f"document_classification_{split}.jsonl",
            document_split,
        )
        _write_jsonl(
            output_directory / f"skill_sequence_labeling_{split}.jsonl",
            skill_split,
        )
    summary_path = output_directory / "package_summary.json"
    summary_path.write_text(
        json.dumps(package.summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_path.chmod(0o600)
