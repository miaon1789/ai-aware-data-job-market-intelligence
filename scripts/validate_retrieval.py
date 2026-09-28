#!/usr/bin/env python3
"""Run retrieval checks, capture an explicit baseline, or verify sealed evidence."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from job_market_intelligence.retrieval.validation import (  # noqa: E402
    SCORING_FILES,
    capture_code_changes,
    check,
    code_snapshot,
    compare_scores,
    digest,
    expected_queries,
    json_digest,
    load_baseline,
    load_policy,
    new_run,
    now,
    overall_status,
    privacy_offenders,
    published_files,
    published_snapshot,
    render_report,
    seal_run,
    test_result,
    validate_scores,
    verify_current_publication,
    verify_run,
    write_json,
)

RAG_TESTS = ["tests/test_rag_routing.py", "tests/test_rag_stats.py"]
QUICK_TESTS = [
    "tests/test_retrieval_evaluation.py",
    "tests/test_golden_set.py",
    "tests/test_retrieval_privacy.py",
    "tests/test_chunking.py",
    *RAG_TESTS,
    "tests/test_validation_workflow.py",
]
HOOK_TESTS = [
    "tests/test_retrieval_evaluation.py",
    "tests/test_golden_set.py",
    "tests/test_retrieval_privacy.py",
    *RAG_TESTS,
    "tests/test_validation_workflow.py::test_privacy_faults_without_private_corpus",
]


def command(
    argv: list[str], root: Path, run: Path, name: str, commands: list[dict], *, timeout: int = 180
) -> int | None:
    record = {"argv": argv, "cwd": str(root), "started_at": now(), "exit_code": None}
    commands.append(record)
    started = time.monotonic()
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    # Deterministic validation must not inherit test-selection or plugin overrides.
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_PLUGINS", None)
    try:
        with (run / f"{name}.log").open("w") as log:
            result = subprocess.run(
                argv,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env=environment,
                check=False,
            )
        record["exit_code"] = result.returncode
    except subprocess.TimeoutExpired:
        record["reason"] = "timeout"
    except OSError as exc:
        record["reason"] = str(exc)
    record["seconds"] = round(time.monotonic() - started, 3)
    return record["exit_code"]


def run_validation(args, root: Path = ROOT) -> tuple[int, dict]:
    import pandas as pd

    run = new_run(root, args.mode)
    targets = HOOK_TESTS if args.profile == "hook" else QUICK_TESTS
    checks, commands, comparisons = [], [], []
    input_paths, input_hashes = {}, {}
    manifest = {
        "schema_version": 1,
        "run_id": run.name,
        "mode": args.mode,
        "profile": args.profile,
        "test_targets": targets,
        "commands": commands,
        "invocation": sys.argv,
        "started_at": now(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        },
    }
    before = None
    baseline_frame = baseline_manifest = None
    scanned_publication = None
    try:
        before = code_snapshot(root)
        manifest["code"] = before
        manifest["evaluator_sha256"] = json_digest(
            {name: before["files"][name] for name in SCORING_FILES}
        )
        capture_code_changes(root, run, before)
        control_paths = {"policy": args.policy}
        if args.config:
            control_paths["config"] = args.config
        manifest["control_files"] = {
            name: {"path": str(path.resolve()), "sha256": digest(path)}
            for name, path in control_paths.items()
        }
        policy = load_policy(args.policy)
        policy_hash = digest(args.policy)
        manifest.update(policy_sha256=policy_hash, policy=policy)
        expected = expected_queries(args.queries, policy)
        input_paths = {"queries": args.queries}
        if args.mode != "quick":
            input_paths.update(
                jobs=args.jobs, database=args.database, document_labels=args.document_labels
            )
        elif args.jobs.is_file():
            input_paths["jobs"] = args.jobs
        missing = [name for name, path in input_paths.items() if not path.is_file()]
        if missing:
            checks.append(
                check(
                    "inputs",
                    "NOT RUN",
                    "required input files are missing: " + ", ".join(missing),
                    missing=missing,
                )
            )
        else:
            input_hashes = {name: digest(path) for name, path in input_paths.items()}
            manifest.update(
                input_hashes=input_hashes,
                input_paths={name: str(path.resolve()) for name, path in input_paths.items()},
            )
            checks.append(
                check(
                    "inputs",
                    "PASS",
                    "query set and input availability checked",
                    query_counts=dict(
                        pd.Series(list(expected.values())).value_counts().astype(int).to_dict()
                    ),
                )
            )

        exit_code = command(
            [
                sys.executable,
                "-m",
                "pytest",
                *targets,
                "-q",
                "-ra",
                "--junitxml=" + str(run / "tests.xml"),
            ],
            root,
            run,
            "tests",
            commands,
            timeout=args.test_timeout,
        )
        checks.append(test_result(run / "tests.xml", exit_code))

        scanned_publication = published_snapshot(root)
        if args.jobs.is_file():
            privacy_jobs_hash = digest(args.jobs)
            jobs = pd.read_csv(args.jobs, dtype=str, keep_default_na=False)
            offenders = privacy_offenders(published_files(root), jobs)
            checks.append(
                check(
                    "private_corpus_privacy",
                    "FAIL" if offenders else "PASS",
                    "known advertisement identifiers or text fragments found in published files"
                    if offenders
                    else "bounded known-id and opening-text scan completed",
                    required=args.mode != "quick",
                    offenders=offenders,
                    corpus_sha256=privacy_jobs_hash,
                )
            )
            if digest(args.jobs) != privacy_jobs_hash:
                checks.append(check("privacy_input_stability", "FAIL", "privacy corpus changed"))
        else:
            checks.append(
                check(
                    "private_corpus_privacy",
                    "NOT RUN",
                    "private corpus unavailable",
                    required=args.mode != "quick",
                )
            )

        if args.mode == "full":
            if args.baseline is None or not args.baseline.is_dir():
                checks.append(check("baseline", "NOT RUN", "explicit baseline run is required"))
            elif not missing:
                baseline_manifest, baseline_frame = load_baseline(
                    args.baseline, input_hashes, policy_hash, manifest["evaluator_sha256"]
                )
                manifest["baseline"] = {
                    "run_id": baseline_manifest["run_id"],
                    "manifest_sha256": digest(args.baseline / "manifest.json"),
                    "path": str(args.baseline.resolve()),
                    "code": baseline_manifest["code"],
                }
                checks.append(check("baseline", "PASS", "baseline provenance checked"))

        can_capture = args.mode != "quick" and overall_status(checks) == "PASS"
        if can_capture:
            from dataclasses import asdict

            from job_market_intelligence.retrieval.validation_benchmark import checked_config

            config = dict(policy["retrieval_config"])
            if args.config:
                config.update(json.loads(args.config.read_text()))
            config = asdict(checked_config(config))
            manifest.update(config=config, config_sha256=json_digest(config))
            config_file_hash = digest(args.config) if args.config else None
            request = {
                "run_id": run.name,
                "config": config,
                "expected_queries": expected,
                "input_paths": manifest["input_paths"],
                "input_hashes": input_hashes,
            }
            write_json(run / "request.json", request)
            exit_code = command(
                [
                    sys.executable,
                    str(root / "scripts/validate_retrieval.py"),
                    "worker",
                    "--run",
                    str(run),
                ],
                root,
                run,
                "benchmark",
                commands,
                timeout=args.benchmark_timeout,
            )
            if exit_code == 2 and (run / "worker_error.json").is_file():
                error = json.loads((run / "worker_error.json").read_text())
                checks.append(
                    check(
                        "benchmark_capture",
                        "NOT RUN",
                        "benchmark dependency unavailable",
                        private_detail=error["reason"],
                    )
                )
            elif exit_code is None:
                checks.append(check("benchmark_capture", "NOT RUN", "benchmark timed out"))
            elif exit_code != 0:
                checks.append(check("benchmark_capture", "FAIL", "benchmark command failed"))
            else:
                scores = pd.read_csv(run / "scores.csv", dtype={"query_id": str})
                validate_scores(scores, expected, policy["metric"])
                captured = json.loads((run / "capture.json").read_text())
                if (
                    captured["run_id"] != run.name
                    or captured["scores_sha256"] != digest(run / "scores.csv")
                    or captured["config_sha256"] != manifest["config_sha256"]
                ):
                    raise ValueError("benchmark output identity or hash mismatch")
                checks.append(check("benchmark_capture", "PASS", "fresh scores captured"))
                if args.mode == "full":
                    verify_run(args.baseline, baseline_manifest["run_id"])
                    if (
                        digest(args.baseline / "manifest.json")
                        != manifest["baseline"]["manifest_sha256"]
                    ):
                        raise ValueError("baseline changed during validation")
                    baseline_capture = json.loads((args.baseline / "capture.json").read_text())
                    if baseline_capture["judgements_sha256"] != captured["judgements_sha256"]:
                        raise ValueError("resolved relevance differs from the baseline")
                    comparisons = compare_scores(baseline_frame, scores, expected, policy)
                    checks.append(
                        check(
                            "real_benchmark",
                            "FAIL" if any(r["status"] == "FAIL" for r in comparisons) else "PASS",
                            "paired comparison completed",
                            effect="SEE_GROUP_RESULTS",
                        )
                    )
            if args.config and digest(args.config) != config_file_hash:
                raise ValueError("configuration changed during validation")
        if digest(args.policy) != policy_hash:
            raise ValueError("policy changed during validation")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        checks.append(
            check(
                "execution",
                "NOT RUN",
                "required file or dependency unavailable",
                private_detail=str(exc),
            )
        )
    except Exception as exc:
        checks.append(
            check(
                "integrity",
                "FAIL",
                "invalid input or inconsistent validation evidence",
                private_detail=f"{type(exc).__name__}: {exc}",
            )
        )
    except KeyboardInterrupt:
        checks.append(check("execution", "NOT RUN", "validation interrupted"))
    finally:
        if not any(c["name"] == "real_benchmark" for c in checks):
            checks.append(
                check(
                    "real_benchmark",
                    "NOT RUN",
                    {
                        "quick": "quick mode does not run a real benchmark comparison",
                        "baseline": (
                            "baseline mode captures reference scores without comparing versions"
                        ),
                    }.get(
                        args.mode, "comparison prerequisites or benchmark capture did not complete"
                    ),
                    required=args.mode == "full",
                    effect="NOT EVALUATED",
                )
            )
        checks.append(
            check(
                "answer_quality",
                "NOT RUN",
                "answer quality has not been measured end to end",
                required=False,
                effect="NOT EVALUATED",
            )
        )
        try:
            after = code_snapshot(root)
            stable = before is not None and before["fingerprint"] == after["fingerprint"]
            stable = stable and before["commit"] == after["commit"]
            stable = stable and all(
                path.is_file() and digest(path) == input_hashes[name]
                for name, path in input_paths.items()
                if name in input_hashes
            )
            stable = stable and all(
                Path(item["path"]).is_file() and digest(Path(item["path"])) == item["sha256"]
                for item in manifest.get("control_files", {}).values()
            )
            checks.append(
                check(
                    "source_stability",
                    "PASS" if stable else "FAIL",
                    "code and hashed inputs unchanged after execution"
                    if stable
                    else "code or hashed inputs changed during execution",
                )
            )
        except Exception as exc:
            checks.append(
                check(
                    "source_stability",
                    "FAIL",
                    "source or input stability could not be checked",
                    private_detail=str(exc),
                )
            )
        if scanned_publication is not None:
            publication_stable = scanned_publication == published_snapshot(root)
            checks.append(
                check(
                    "privacy_scope_stability",
                    "PASS" if publication_stable else "FAIL",
                    "scanned file set and content unchanged"
                    if publication_stable
                    else "scanned files were added, deleted or changed",
                )
            )
        write_json(run / "checks.json", checks)
        write_json(run / "comparisons.json", comparisons)
        report = render_report(manifest, checks, comparisons)
        (run / "report.md").write_text(report)
        manifest["status"] = overall_status(checks)
        # Only allowlisted fields leave the private run directory. Reasons are controlled
        # text. Raw exception details, logs, private paths and advertisement IDs stay private.
        public = root / "reports/validation" / run.name
        staged = run / "export"
        staged.mkdir(exist_ok=False)
        write_json(
            staged / "summary.json",
            {
                "run_id": run.name,
                "mode": args.mode,
                "profile": args.profile,
                "test_targets": targets,
                "status": manifest["status"],
                "code_commit": manifest.get("code", {}).get("commit"),
                "code_dirty": manifest.get("code", {}).get("dirty"),
                "code_fingerprint": manifest.get("code", {}).get("fingerprint"),
                "policy_sha256": manifest.get("policy_sha256"),
                "evaluator_sha256": manifest.get("evaluator_sha256"),
                "checks": [
                    {key: c[key] for key in ("name", "status", "required", "reason")}
                    | ({"counts": c["counts"]} if "counts" in c else {})
                    | ({"effect": c["effect"]} if "effect" in c else {})
                    for c in checks
                ],
                "comparisons": comparisons,
            },
        )
        (staged / "report.md").write_text(report)
        manifest["published_files"] = {
            **(scanned_publication or {}),
            str((public / "summary.json").relative_to(root)): digest(staged / "summary.json"),
            str((public / "report.md").relative_to(root)): digest(staged / "report.md"),
        }
        # Exports contain only generated aggregate fields. Bind them to this run's scan
        # without importing their prose back into the agent or exposing private logs.
        seal_run(run, manifest)
        verify_run(run, run.name)
        # Publish the two-file bundle atomically after validation. Keep the sealed
        # originals private so a failed export cannot invalidate their hashes.
        import shutil

        pending = run.parent / (".publish-" + run.name)
        shutil.copytree(staged, pending)
        public.parent.mkdir(parents=True, exist_ok=True)
        if public.exists():
            raise ValueError("public run directory already exists")
        pending.rename(public)
    status = manifest["status"]
    return {"PASS": 0, "FAIL": 1, "NOT RUN": 2}[status], {
        "run_id": run.name,
        "status": status,
        "run_dir": str(run),
        "report": str(run / "report.md"),
        "public_report": str(public / "report.md"),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("quick", "full", "baseline", "verify", "worker"))
    result.add_argument("--profile", choices=("quick", "hook"), default="quick")
    result.add_argument("--policy", type=Path, default=ROOT / "eval/validation_policy.json")
    result.add_argument("--queries", type=Path, default=ROOT / "eval/golden/queries.jsonl")
    result.add_argument("--jobs", type=Path, default=ROOT / "data/private/retrieval/jobs_clean.csv")
    result.add_argument(
        "--database", type=Path, default=ROOT / "data/private/retrieval/job_market.duckdb"
    )
    result.add_argument(
        "--document-labels",
        type=Path,
        default=ROOT
        / "data/private/annotations_expanded/processed_adjudicated/document_labels.csv",
    )
    result.add_argument(
        "--config", type=Path, help="JSON overrides for the retrieval configuration"
    )
    result.add_argument("--baseline", type=Path)
    result.add_argument("--run", type=Path)
    result.add_argument("--expected-run-id")
    result.add_argument(
        "--current", action="store_true", help="also verify current code and inputs"
    )
    result.add_argument("--test-timeout", type=int, default=180)
    result.add_argument("--benchmark-timeout", type=int, default=3600)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.mode in {"worker", "verify"}:
        private_runs = ROOT / "data/private/validation"
        if args.run is None or not args.run.resolve().is_relative_to(private_runs):
            raise ValueError("run must be inside data/private/validation")
        if args.mode == "worker":
            from job_market_intelligence.retrieval.validation_benchmark import worker

            return worker(args.run)
        if not args.expected_run_id:
            raise ValueError("verify requires --expected-run-id from the original command output")
        manifest = verify_run(args.run, args.expected_run_id)
        if args.current:
            current = code_snapshot(ROOT)
            if (
                current["fingerprint"] != manifest["code"]["fingerprint"]
                or current["commit"] != manifest["code"]["commit"]
            ):
                raise ValueError("evidence does not describe the current code")
            for name, sha in manifest.get("input_hashes", {}).items():
                if digest(Path(manifest["input_paths"][name])) != sha:
                    raise ValueError("evidence does not describe current inputs")
            for item in manifest.get("control_files", {}).values():
                if digest(Path(item["path"])) != item["sha256"]:
                    raise ValueError("evidence does not describe the current policy or config")
            verify_current_publication(ROOT, manifest)
        print(
            json.dumps(
                {
                    "run_id": args.run.name,
                    "evidence": "VERIFIED",
                    "check_status": manifest["status"],
                }
            )
        )
        return 0
    if args.profile == "hook" and args.mode != "quick":
        raise ValueError("hooks can only run quick validation")
    code, summary = run_validation(args)
    print(json.dumps(summary))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
