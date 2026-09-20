"""Opt-in integration checks against real dbt execution, not SQL string matching."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from job_market_intelligence.ingestion import load_job_ads
from job_market_intelligence.preprocessing import deduplicate_job_ads
from job_market_intelligence.seniority import classify_seniority
from job_market_intelligence.skills import extract_mentions_for_jobs
from scripts.generate_synthetic_data import generate_csv
from scripts.prepare_dbt import prepare

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DBT_TESTS") != "1", reason="Set RUN_DBT_TESTS=1 in the dbt environment"
)


def test_dbt_parity_and_failure_detection():
    """Check no-mention denominators, repeated mentions and genuine failing tests."""
    executable = Path(sys.executable).with_name("dbt")
    assert executable.exists(), "Install dbt/requirements.txt in this Python environment"
    parent = ROOT / "data/private/dbt"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test-", dir=parent) as directory:
        work = Path(directory)
        jobs, _ = deduplicate_job_ads(load_job_ads(generate_csv(work / "fixture.csv")))
        jobs["analysis_role"] = jobs["role_label"]
        jobs["seniority"] = jobs.apply(
            lambda row: classify_seniority(row["title"], row["description"])[0], axis=1
        )
        mentions = extract_mentions_for_jobs(jobs)
        # Repeated mentions must not inflate distinct-ad counts.
        mentions = pd.concat([mentions, mentions.iloc[[0]]], ignore_index=True)
        # A retained ad with no skill mentions must still contribute to the denominator.
        extra = jobs.iloc[[0]].copy()
        extra["job_id"] = "SYN-NO-SKILLS"
        jobs = pd.concat([jobs, extra], ignore_index=True)
        jobs.to_csv(work / "jobs_clean.csv", index=False)
        mentions.to_csv(work / "skill_mentions.csv", index=False)
        db = work / "test.duckdb"
        prepare(work, db)
        env = {
            **os.environ,
            "DBT_DUCKDB_PATH": str(db),
            "DBT_SEND_ANONYMOUS_USAGE_STATS": "false",
        }
        project = work / "project"
        shutil.copytree(
            ROOT / "dbt",
            project,
            ignore=shutil.ignore_patterns("target", "logs", "dbt_packages", ".user.yml"),
        )

        def run(*args):
            return subprocess.run(
                [
                    str(executable),
                    *args,
                    "--project-dir",
                    str(project),
                    "--profiles-dir",
                    str(project),
                ],
                env=env,
                text=True,
                capture_output=True,
                timeout=180,
                check=False,
            )

        result = run("build")
        assert result.returncode == 0, result.stdout + result.stderr
        with duckdb.connect(str(db)) as con:
            actual = con.execute("select sum(job_count) from analytics.mart_role_city").fetchone()[
                0
            ]
            assert actual == len(jobs)
            assert con.execute("select count(*) from analytics.stg_skill_mentions").fetchone()[
                0
            ] < len(mentions)
            row = con.execute("select * from analytics.mart_role_skill limit 1").fetchdf().iloc[0]
            denominator = len(jobs[(jobs.city == row.city) & (jobs.analysis_role == row.role)])
            assert row.share_of_ads == pytest.approx(row.ads_with_skill / denominator)
        docs = run("docs", "generate")
        assert docs.returncode == 0, docs.stdout + docs.stderr
        assert (project / "target/catalog.json").exists()

        # Deliberately corrupt the isolated input, never the actual analysis database.
        with duckdb.connect(str(db)) as con:
            con.execute("insert into jobs select * from jobs limit 1")
            con.execute("update jobs set city = '' where job_id = 'SYN-NO-SKILLS'")
            con.execute(
                "insert into skill_mentions values ('MISSING-AD', 'SQL', 'test', 'invalid')"
            )
        result = run("run")
        assert result.returncode == 0, result.stdout + result.stderr
        result = run("test")
        assert result.returncode == 1, result.stdout + result.stderr
        results = json.loads((project / "target/run_results.json").read_text())["results"]
        failed = [r["unique_id"] for r in results if r["status"] == "fail"]
        for expected in [
            "source_unique_pipeline_jobs_job_id",
            "not_null_stg_jobs_city",
            "accepted_values_stg_skill_mentions_context",
            "relationships_stg_skill_mentions_job_id",
        ]:
            assert any(expected in test for test in failed), (expected, failed)
        result = run("build")
        assert result.returncode == 1, result.stdout + result.stderr
        results = json.loads((project / "target/run_results.json").read_text())["results"]
        # No descendant mart should be rebuilt using inputs that failed the quality gates.
        assert all(r["status"] == "skipped" for r in results if ".mart_" in r["unique_id"])
