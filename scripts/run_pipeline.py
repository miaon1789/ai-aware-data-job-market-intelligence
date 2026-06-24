#!/usr/bin/env python3
"""Run the complete MVP pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from job_market_intelligence.analytics import build_database  # noqa: E402
from job_market_intelligence.ingestion import load_job_ads  # noqa: E402
from job_market_intelligence.modelling import (  # noqa: E402
    combine_text,
    save_model,
    train_and_evaluate,
)
from job_market_intelligence.preprocessing import deduplicate_job_ads  # noqa: E402
from job_market_intelligence.quality import (  # noqa: E402
    build_quality_report,
    filter_recent_ads,
)
from job_market_intelligence.seniority import classify_seniority  # noqa: E402
from job_market_intelligence.skills import extract_mentions_for_jobs  # noqa: E402
from scripts.generate_synthetic_data import generate_csv  # noqa: E402


def report_path(path: Path) -> str:
    """Avoid writing developer-specific absolute paths into public reports."""

    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def run(
    input_path: Path,
    training_path: Path,
    output_dir: Path,
    reports_dir: Path,
    max_age_days: int = 90,
) -> dict[str, object]:
    jobs = load_job_ads(input_path)
    original_rows = len(jobs)
    jobs, age_filtered_rows, reference_date = filter_recent_ads(
        jobs, max_age_days=max_age_days
    )
    jobs, duplicate_audit = deduplicate_job_ads(jobs)

    seniority = jobs.apply(
        lambda row: classify_seniority(row["title"], row["description"]), axis=1
    )
    jobs[["seniority", "seniority_evidence"]] = list(seniority)
    skill_mentions = extract_mentions_for_jobs(jobs)

    if input_path.resolve() == training_path.resolve():
        training_jobs = jobs
    else:
        training_jobs, _ = deduplicate_job_ads(load_job_ads(training_path))
    model_result = train_and_evaluate(training_jobs)
    model_result.metrics["training_dataset"] = report_path(training_path)
    jobs["predicted_role"] = model_result.model.predict(combine_text(jobs))
    jobs["analysis_role"] = jobs["role_label"].where(
        jobs["role_label"].notna() & jobs["role_label"].ne(""), jobs["predicted_role"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(output_dir / "jobs_clean.csv", index=False)
    skill_mentions.to_csv(output_dir / "skill_mentions.csv", index=False)
    duplicate_audit.to_csv(output_dir / "duplicate_audit.csv", index=False)
    model_result.predictions.to_csv(output_dir / "cross_validation_predictions.csv", index=False)
    save_model(model_result.model, output_dir / "role_classifier.joblib")

    metrics_path = reports_dir / "model_metrics.json"
    metrics_path.write_text(json.dumps(model_result.metrics, indent=2), encoding="utf-8")
    quality_report = build_quality_report(
        jobs,
        skill_mentions,
        duplicate_audit,
        original_rows=original_rows,
        age_filtered_rows=age_filtered_rows,
        max_age_days=max_age_days,
        reference_date=reference_date,
    )
    (reports_dir / "data_quality.json").write_text(
        json.dumps(quality_report, indent=2), encoding="utf-8"
    )
    build_database(
        output_dir / "job_market.duckdb", jobs, skill_mentions, duplicate_audit
    )

    return {
        "input_rows": original_rows,
        "age_filtered_rows": age_filtered_rows,
        "age_window_days": max_age_days,
        "clean_rows": len(jobs),
        "duplicates_removed": len(duplicate_audit),
        "skill_mentions": len(skill_mentions),
        "training_rows": len(training_jobs),
        "macro_f1": model_result.metrics["macro_f1"],
        "database": str(output_dir / "job_market.duckdb"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Permitted CSV matching docs/DATA_CARD.md")
    parser.add_argument(
        "--training-input",
        type=Path,
        help="Labelled CSV for model training; defaults to the synthetic fixture",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=90,
        help="Retain ads posted within this many days of retrieval/dataset reference date",
    )
    args = parser.parse_args()

    synthetic_path = ROOT / "data/synthetic/job_ads.csv"
    generate_csv(synthetic_path)
    input_path = args.input or synthetic_path
    training_path = args.training_input or synthetic_path
    summary = run(
        input_path,
        training_path,
        args.output_dir,
        args.reports_dir,
        max_age_days=args.max_age_days,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
