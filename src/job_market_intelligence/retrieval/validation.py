"""Deterministic validation and evidence handling shared by the CLI and hooks."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .evaluation import metric_columns, paired_comparison
from .golden import QUERY_CLASSES, load_queries

STATUSES = {"PASS", "FAIL", "NOT RUN"}
SCORING_FILES = (
    "src/job_market_intelligence/retrieval/evaluation.py",
    "src/job_market_intelligence/retrieval/golden.py",
    "src/job_market_intelligence/retrieval/validation_benchmark.py",
)
OPTIONAL_PRIVATE_TESTS = {
    "test_published_artefacts_contain_no_advertisement_ids",
    "test_published_artefacts_contain_no_advertisement_text",
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Replace one JSON file atomically. Completion is sealed separately."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def check(name: str, status: str, reason: str, *, required: bool = True, **details) -> dict:
    if status not in STATUSES:
        raise ValueError("unknown check status")
    return {"name": name, "status": status, "reason": reason, "required": required, **details}


def overall_status(checks: list[dict]) -> str:
    if any(item["status"] == "FAIL" for item in checks):
        return "FAIL"
    required = [item for item in checks if item["required"]]
    if not required or any(item["status"] == "NOT RUN" for item in required):
        return "NOT RUN"
    return "PASS"


def new_run(root: Path, mode: str) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:12]
    path = root / "data/private/validation" / run_id
    path.mkdir(parents=True, mode=0o700, exist_ok=False)
    path.chmod(0o700)
    write_json(path / "started.json", {"run_id": run_id, "mode": mode, "started_at": now()})
    return path


def code_snapshot(root: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root).decode().strip()

    names = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0")
    prefixes = ("src/", "scripts/", "tests/", "eval/", "config/", ".claude/", ".github/", "docs/")
    fixed = {"README.md", "pyproject.toml", "Makefile", ".gitignore", ".dockerignore", "Dockerfile"}
    paths = sorted(name for name in set(names) if name.startswith(prefixes) or name in fixed)
    files = {name: digest(root / name) for name in paths if (root / name).is_file()}
    working_tree = git("status", "--short", "--untracked-files=all")
    untracked = sorted(
        filter(None, git("ls-files", "-z", "--others", "--exclude-standard").split("\0"))
    )
    return {
        "commit": git("rev-parse", "HEAD"),
        "working_tree": working_tree,
        "dirty": bool(working_tree),
        "diff_summary": git("diff", "HEAD", "--stat"),
        "diff_summary_scope": "tracked_only",
        "untracked_files": untracked,
        "untracked_file_count": len(untracked),
        "paths": paths,
        "files": files,
        "fingerprint": json_digest(files),
    }


def capture_code_changes(root: Path, run: Path, snapshot: dict) -> None:
    """Keep scoped diffs and untracked source copies as private, sealed review evidence."""
    folder = run / "code_changes"
    folder.mkdir()
    paths = snapshot["paths"]
    patch = (
        subprocess.check_output(
            ["git", "diff", "--no-ext-diff", "--no-textconv", "--no-renames", "HEAD", "--", *paths],
            cwd=root,
        )
        if paths
        else b""
    )
    (folder / "tracked.patch").write_bytes(patch)
    for name in snapshot["untracked_files"]:
        if name not in snapshot["files"]:
            continue
        target = folder / "untracked" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        if digest(target) != snapshot["files"][name]:
            raise ValueError("untracked source changed while capturing review evidence")


def load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1 or policy.get("metric") != "ndcg@10":
        raise ValueError("validation policy must use schema 1 and ndcg@10")
    if policy.get("direction") != "higher_is_better":
        raise ValueError("unsupported metric direction")
    if sorted(policy.get("query_classes", [])) != sorted(QUERY_CLASSES):
        raise ValueError("policy must contain all four query classes exactly once")
    if policy.get("multiple_comparisons") != "bonferroni":
        raise ValueError("this policy requires Bonferroni adjustment")
    alpha = policy.get("family_alpha", 0)
    if not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("family_alpha must be between zero and one")
    for key, minimum in (("bootstrap_iterations", 100), ("minimum_queries_per_class", 2)):
        if type(policy.get(key)) is not int or policy[key] < minimum:
            raise ValueError(f"invalid {key}")
    if type(policy.get("seed")) is not int or policy["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    return policy


def expected_queries(path: Path, policy: dict) -> dict[str, str]:
    golden = load_queries(path)
    expected = {query.query_id: query.query_class for query in golden}
    if not expected or any(not key.strip() for key in expected):
        raise ValueError("query set must contain nonempty ids")
    counts = Counter(expected.values())
    for query_class in policy["query_classes"]:
        if counts[query_class] < policy["minimum_queries_per_class"]:
            raise ValueError(f"too few queries in class {query_class}")
    return expected


def validate_scores(frame: pd.DataFrame, expected: dict[str, str], metric: str) -> None:
    if not {"query_id", "query_class", metric}.issubset(frame.columns):
        raise ValueError("scores are missing required columns")
    if frame.empty or frame.query_id.isna().any() or frame.query_id.duplicated().any():
        raise ValueError("scores require unique, nonempty query ids")
    if set(frame.query_id) != set(expected):
        raise ValueError("score query ids do not exactly match the expected query set")
    if not frame.query_class.eq(frame.query_id.map(expected)).all():
        raise ValueError("score query classes do not match the query set")
    values = frame[metric_columns(frame)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("retrieval metrics must be finite values between zero and one")


def compare_scores(
    baseline: pd.DataFrame, candidate: pd.DataFrame, expected: dict[str, str], policy: dict
) -> list[dict]:
    """One predeclared family: overall plus four classes, candidate minus baseline."""
    metric = policy["metric"]
    for frame in (baseline, candidate):
        validate_scores(frame, expected, metric)
    combined = pd.concat(
        [baseline.assign(config="baseline"), candidate.assign(config="candidate")],
        ignore_index=True,
    )
    alpha = policy["family_alpha"] / (1 + len(policy["query_classes"]))
    options = {
        "metric": metric,
        "alpha": alpha,
        "seed": policy["seed"],
        "iterations": policy["bootstrap_iterations"],
    }
    overall = paired_comparison(combined, "candidate", "baseline", **options)
    overall["query_class"] = "overall"
    groups = paired_comparison(
        combined, "candidate", "baseline", group_columns=["query_class"], **options
    )
    results = pd.concat([overall, groups], ignore_index=True).to_dict(orient="records")
    for row in results:
        low, high = row["difference_lo"], row["difference_hi"]
        row["effect"] = "REGRESSION" if high < 0 else "IMPROVEMENT" if low > 0 else "INCONCLUSIVE"
        row["status"] = "FAIL" if row["effect"] == "REGRESSION" else "PASS"
        row["interval_confidence"] = 1 - alpha
        row["multiple_comparisons"] = "bonferroni"
    return results


def test_result(xml_path: Path, exit_code: int | None) -> dict:
    """Count actual test cases, including skipped cases, rather than trusting exit 0."""
    if exit_code is None:
        return check(
            "tests",
            "NOT RUN",
            "test command did not complete",
            exit_code=None,
            counts_complete=False,
        )
    if not xml_path.is_file():
        return check("tests", "FAIL", "test report is missing", exit_code=exit_code)
    cases = list(ET.parse(xml_path).getroot().iter("testcase"))
    counts = {
        "collected": len(cases),
        "executed": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
    }
    unexpected_skips = []
    for case in cases:
        if case.find("skipped") is not None:
            counts["skipped"] += 1
            if case.get("name") not in OPTIONAL_PRIVATE_TESTS:
                unexpected_skips.append(case.get("name"))
            continue
        counts["executed"] += 1
        if case.find("failure") is not None:
            counts["failed"] += 1
        elif case.find("error") is not None:
            counts["errors"] += 1
        else:
            counts["passed"] += 1
    if exit_code != 0 or not counts["collected"] or counts["failed"] or counts["errors"]:
        status, reason = "FAIL", "test command failed or collected no tests"
    elif not counts["executed"] or unexpected_skips:
        status, reason = "NOT RUN", "required tests did not execute"
    else:
        status, reason = "PASS", "selected tests completed"
    return check(
        "tests",
        status,
        reason,
        counts=counts,
        exit_code=exit_code,
        unexpected_skips=unexpected_skips,
    )


def published_files(root: Path) -> list[Path]:
    return [
        path
        for folder in ("eval", "reports/retrieval", "reports/validation")
        for path in (root / folder).rglob("*")
        if path.is_file()
    ]


def published_snapshot(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): digest(path) for path in published_files(root)}


def verify_current_publication(root: Path, manifest: dict) -> None:
    if manifest.get("published_files") != published_snapshot(root):
        raise ValueError("published file set or content changed after the privacy check")


def privacy_offenders(paths: list[Path], jobs: pd.DataFrame) -> list[str]:
    """Known-id and normalized text-fragment scan. This is not a complete DLP system."""
    if jobs.empty or "job_id" not in jobs or "description" not in jobs:
        raise ValueError("privacy scan requires a nonempty corpus with ids and descriptions")
    ids = [str(value) for value in jobs.job_id if str(value)]
    pattern = re.compile(r"(?<![\w-])(?:" + "|".join(map(re.escape, ids)) + r")(?![\w-])")
    fragments = [
        " ".join(str(text).split()[:8]) for text in jobs.description if len(str(text).split()) >= 8
    ]
    offenders = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        normalized = " ".join(text.split())
        if pattern.search(text) or any(fragment in normalized for fragment in fragments):
            offenders.append(str(path))  # Never copy the private content into diagnostics.
    return offenders


def seal_run(run: Path, manifest: dict) -> None:
    manifest["finished_at"] = now()
    manifest["state"] = "completed"
    manifest["artifacts"] = {
        str(path.relative_to(run)): digest(path)
        for path in sorted(run.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "complete.json", "review.md"}
    }
    write_json(run / "manifest.json", manifest)
    write_json(
        run / "complete.json",
        {"run_id": manifest["run_id"], "manifest_sha256": digest(run / "manifest.json")},
    )


def verify_run(run: Path, expected_run_id: str | None = None) -> dict:
    """Check a sealed run before accepting its files. Legacy reports cannot be imported."""
    try:
        marker = json.loads((run / "complete.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("run is missing a valid manifest or completion marker") from exc
    run_id = expected_run_id or run.name
    if not run_id == run.name == manifest.get("run_id") == marker.get("run_id"):
        raise ValueError("run identity mismatch: stale or copied evidence")
    if marker.get("manifest_sha256") != digest(run / "manifest.json"):
        raise ValueError("manifest hash mismatch")
    if manifest.get("schema_version") != 1 or manifest.get("state") != "completed":
        raise ValueError("unsupported or incomplete run")
    artifacts = manifest.get("artifacts", {})
    if not {"started.json", "checks.json", "report.md"}.issubset(artifacts):
        raise ValueError("required run evidence is missing")
    for name, sha in artifacts.items():
        path = (run / name).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file() or digest(path) != sha:
            raise ValueError(f"artifact missing or hash mismatch: {name}")
    if manifest.get("status") != overall_status(json.loads((run / "checks.json").read_text())):
        raise ValueError("manifest status does not match check evidence")
    return manifest


def load_baseline(
    run: Path, input_hashes: dict, policy_hash: str, evaluator_hash: str
) -> tuple[dict, pd.DataFrame]:
    manifest = verify_run(run)
    if manifest.get("mode") != "baseline" or manifest.get("status") != "PASS":
        raise ValueError("baseline must be an explicitly captured, successful baseline run")
    if manifest.get("input_hashes") != input_hashes or manifest.get("policy_sha256") != policy_hash:
        raise ValueError("baseline inputs or policy differ from this run")
    if manifest.get("evaluator_sha256") != evaluator_hash:
        raise ValueError("baseline scoring implementation differs from this run")
    if "scores.csv" not in manifest["artifacts"] or "capture.json" not in manifest["artifacts"]:
        raise ValueError("baseline has no benchmark capture")
    return manifest, pd.read_csv(run / "scores.csv", dtype={"query_id": str})


def render_report(manifest: dict, checks: list[dict], comparisons: list[dict]) -> str:
    code = manifest.get("code", {})
    tree_state = {True: "contains uncommitted changes", False: "clean"}.get(
        code.get("dirty"), "unavailable"
    )
    lines = [
        "# Retrieval validation",
        "",
        f"Run: {manifest['run_id']}",
        f"Mode: {manifest['mode']}",
        f"Check status: {overall_status(checks)}",
        "",
        f"Code commit: {code.get('commit', 'unavailable')}",
        f"Working tree: {tree_state}",
        f"Source fingerprint (SHA256): {code.get('fingerprint', 'unavailable')}",
        f"Policy SHA256: {manifest.get('policy_sha256', 'unavailable')}",
        f"Evaluator SHA256: {manifest.get('evaluator_sha256', 'unavailable')}",
        "",
        "PASS means completed without a preset failure condition. It does not prove",
        "equivalence, absence of regression, or model safety.",
        "",
        "| Check | Status | Required |",
        "|---|---|---|",
    ]
    lines.extend(f"| {c['name']} | {c['status']} | {c['required']} |" for c in checks)
    for item in checks:
        if "counts" in item:
            counts = item["counts"]
            lines += ["", "Tests: " + ", ".join(f"{k}={v}" for k, v in counts.items()) + "."]
    lines += [
        "",
        "Machine checks and the optional agent review are separate.",
        "The reviewer cannot change machine status.",
        "",
    ]
    if comparisons:
        confidence = comparisons[0]["interval_confidence"]
        family_alpha = manifest["policy"]["family_alpha"]
        lines += [
            "Candidate minus baseline. Bonferroni adjustment over five comparisons.",
            f"Family alpha: {family_alpha:g}. Individual confidence: {confidence:.1%}.",
            "Percentile bootstrap coverage is approximate. Values below are for display.",
            "Use comparisons.json for full precision.",
            "",
            "| Group | Queries | Difference | Interval | Status | Effect |",
            "|---|---:|---:|---|---|---|",
        ]
        for row in comparisons:
            lines.append(
                f"| {row['query_class']} | {row['queries']} | {row['mean_difference']:.6g} | "
                f"[{row['difference_lo']:.6g}, {row['difference_hi']:.6g}] | "
                f"{row['status']} | {row['effect']} |"
            )
        lines += [
            "",
            "INCONCLUSIVE means no clear difference was detected.",
            "Aggregation scores measure retrieval, not the correctness of counted answers.",
        ]
    else:
        lines += ["Real benchmark comparison: NOT RUN. Effect: NOT EVALUATED."]
    return "\n".join(lines) + "\n"
