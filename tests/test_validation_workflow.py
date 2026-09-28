"""Fault injection for evidence handling and the Claude Code adapters."""

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from job_market_intelligence.retrieval.evaluation import paired_comparison
from job_market_intelligence.retrieval.validation import (
    capture_code_changes,
    code_snapshot,
    compare_scores,
    digest,
    json_digest,
    load_baseline,
    load_policy,
    overall_status,
    privacy_offenders,
    validate_scores,
    verify_current_publication,
    verify_run,
    write_json,
)
from job_market_intelligence.retrieval.validation import test_result as read_test_result
from scripts import validate_retrieval as runner
from scripts.hooks import retrieval_feedback as hook

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy():
    value = load_policy(ROOT / "eval/validation_policy.json")
    value["bootstrap_iterations"] = 500
    return value


@pytest.fixture
def expected(policy):
    return {f"{group}-{i}": group for group in policy["query_classes"] for i in range(2)}


def scores(expected, value=0.5):
    return pd.DataFrame(
        [
            {"query_id": key, "query_class": group, "ndcg@10": value}
            for key, group in expected.items()
        ]
    )


def test_small_paired_regression_fails_without_an_eight_point_gate(expected, policy):
    results = compare_scores(scores(expected), scores(expected, 0.49), expected, policy)
    assert len(results) == 5
    assert all(r["status"] == "FAIL" and r["effect"] == "REGRESSION" for r in results)
    assert all(r["interval_confidence"] == 0.99 for r in results)


def test_subgroup_regression_is_not_hidden_by_overall_gain(expected, policy):
    candidate = scores(expected, 0.6)
    candidate.loc[candidate.query_class.eq("mixed"), "ndcg@10"] = 0.49
    rows = {
        r["query_class"]: r for r in compare_scores(scores(expected), candidate, expected, policy)
    }
    assert rows["overall"]["mean_difference"] > 0
    assert rows["mixed"]["status"] == "FAIL"


def test_interval_containing_zero_does_not_claim_improvement(expected, policy):
    candidate = scores(expected)
    candidate["ndcg@10"] += np.tile([-0.01, 0.01], len(candidate) // 2)
    rows = compare_scores(scores(expected), candidate, expected, policy)
    assert all(r["effect"] == "INCONCLUSIVE" and r["status"] == "PASS" for r in rows)


def test_zero_boundary_and_identical_scores_are_inconclusive(expected, policy):
    rows = compare_scores(scores(expected), scores(expected), expected, policy)
    assert all(r["effect"] == "INCONCLUSIVE" for r in rows)


def test_normal_positive_case_passes(expected, policy):
    rows = compare_scores(scores(expected), scores(expected, 0.7), expected, policy)
    assert all(r["effect"] == "IMPROVEMENT" and r["status"] == "PASS" for r in rows)


def test_tiny_regression_keeps_unrounded_direction(expected, policy):
    rows = compare_scores(scores(expected), scores(expected, 0.49999), expected, policy)
    assert all(r["difference_hi"] < 0 and r["effect"] == "REGRESSION" for r in rows)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nan", "inf", "class", "range"])
def test_bad_scores_are_rejected(fault, expected):
    frame = scores(expected)
    if fault == "missing":
        frame = frame.iloc[:-1]
    elif fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif fault == "class":
        frame.loc[0, "query_class"] = "mixed"
    else:
        frame.loc[0, "ndcg@10"] = {"nan": np.nan, "inf": np.inf, "range": 1.1}[fault]
    with pytest.raises(ValueError):
        validate_scores(frame, expected, "ndcg@10")


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nan", "class"])
def test_low_level_pairing_also_rejects_invalid_input(fault, expected):
    left, right = scores(expected).assign(config="a"), scores(expected).assign(config="b")
    if fault == "missing":
        right = right.iloc[:-1]
    elif fault == "duplicate":
        left = pd.concat([left, left.iloc[:1]], ignore_index=True)
    elif fault == "class":
        right.loc[0, "query_class"] = "mixed"
    else:
        right.loc[0, "ndcg@10"] = np.nan
    with pytest.raises(ValueError):
        paired_comparison(pd.concat([left, right]), "a", "b", group_columns=["query_class"])


@pytest.mark.parametrize(
    ("body", "exit_code", "status"),
    [
        ("", 0, "FAIL"),
        ('<testcase name="x"><skipped/></testcase>', 0, "NOT RUN"),
        ('<testcase name="x"><failure/></testcase>', 0, "FAIL"),
        ('<testcase name="x"/>', 1, "FAIL"),
        ('<testcase name="x"/>', 0, "PASS"),
    ],
)
def test_zero_tests_skips_failures_and_exit_codes(tmp_path, body, exit_code, status):
    path = tmp_path / "tests.xml"
    path.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>")
    result = read_test_result(path, exit_code)
    assert result["status"] == status


@pytest.mark.parametrize(
    "report", [None, "<testsuite><testcase name='x'/></testsuite>", "<partial"]
)
def test_inner_timeout_is_incomplete_even_with_a_partial_junit_report(tmp_path, report):
    path = tmp_path / "tests.xml"
    if report is not None:
        path.write_text(report)
    result = read_test_result(path, None)
    assert result["status"] == "NOT RUN" and result["counts_complete"] is False


def test_privacy_faults_without_private_corpus(tmp_path):
    jobs = pd.DataFrame(
        [
            {
                "job_id": "private-fixture-ad-42",
                "description": (
                    "Our fictional hiring team needs someone to maintain research systems."
                ),
            }
        ]
    )
    path = tmp_path / "report.md"
    path.write_text("All aggregate scores passed.")
    assert privacy_offenders([path], jobs) == []
    for leak in ("private-fixture-ad-42", "Our fictional hiring team needs someone to maintain"):
        path.write_text(leak)
        assert privacy_offenders([path], jobs) == [str(path)]
        assert leak not in repr(privacy_offenders([path], jobs))


@pytest.fixture
def snapshot_repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("Reference overview\n")
    (tmp_path / "docs/RETRIEVAL_VALIDATION.md").write_text("Reference policy explanation\n")
    (tmp_path / "docs/removed.md").write_text("Obsolete explanation\n")
    (tmp_path / ".gitignore").write_text(".env\ndocs/private.md\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Validation Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Fixture reference",
        ],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


@pytest.mark.parametrize("name", ["README.md", "docs/RETRIEVAL_VALIDATION.md"])
def test_document_edits_change_the_source_fingerprint(snapshot_repo, name):
    before = code_snapshot(snapshot_repo)
    assert not before["dirty"]
    (snapshot_repo / name).write_text("Changed statistical claim\n")
    after = code_snapshot(snapshot_repo)
    assert after["dirty"] and before["fingerprint"] != after["fingerprint"]
    assert before["files"][name] != after["files"][name]


def test_snapshot_preserves_tracked_and_untracked_review_evidence(snapshot_repo, tmp_path):
    root = snapshot_repo
    (root / "README.md").write_text("Candidate overview\n")
    (root / "docs/removed.md").unlink()
    (root / "docs/new.md").write_text("New explanation\n")
    (root / "notes.txt").write_text("Outside the source snapshot scope\n")
    (root / ".env").write_text("PRIVATE=fixture\n")
    (root / "docs/private.md").write_text("Ignored local draft\n")
    snapshot = code_snapshot(root)
    assert snapshot["diff_summary_scope"] == "tracked_only"
    assert snapshot["untracked_file_count"] == 2
    assert snapshot["untracked_files"] == ["docs/new.md", "notes.txt"]
    assert "?? docs/new.md" in snapshot["working_tree"]
    assert "docs/private.md" not in snapshot["files"]
    run = tmp_path / "run"
    run.mkdir()
    capture_code_changes(root, run, snapshot)
    patch = (run / "code_changes/tracked.patch").read_text()
    assert "+Candidate overview" in patch and "-Reference overview" in patch
    assert "-Obsolete explanation" in patch
    assert (run / "code_changes/untracked/docs/new.md").read_text() == "New explanation\n"
    assert not (run / "code_changes/untracked/notes.txt").exists()
    assert not (run / "code_changes/untracked/.env").exists()


@pytest.fixture
def run_fixture(tmp_path, monkeypatch, policy, expected):
    root = tmp_path
    (root / "eval").mkdir()
    policy_path = root / "eval/policy.json"
    write_json(policy_path, policy)
    queries = root / "eval/queries.jsonl"
    queries.write_text(
        "\n".join(
            json.dumps(
                {
                    "query_id": key,
                    "query_class": group,
                    "query": "Find roles in Melbourne",
                    "relevance": {
                        "type": "metadata_match",
                        "column": "city",
                        "values": ["Melbourne"],
                    },
                }
            )
            for key, group in expected.items()
        )
    )
    jobs = root / "jobs.csv"
    pd.DataFrame(
        [{"job_id": "fixture-private-ad", "description": "Fictional advertisement"}]
    ).to_csv(jobs, index=False)
    database, labels = root / "data.duckdb", root / "labels.csv"
    database.write_bytes(b"synthetic input")
    labels.write_text("job_id\nfixture-private-ad\n")
    args = SimpleNamespace(
        mode="quick",
        profile="quick",
        policy=policy_path,
        queries=queries,
        jobs=jobs,
        database=database,
        document_labels=labels,
        baseline=None,
        config=None,
        test_timeout=30,
        benchmark_timeout=30,
    )
    state = {
        "score": 0.5,
        "test_exit": 0,
        "test_body": '<testcase name="example"/>',
        "benchmark_exit": 0,
    }
    monkeypatch.setattr(
        runner,
        "code_snapshot",
        lambda root: {
            "commit": "synthetic-code",
            "fingerprint": "synthetic-fingerprint",
            "files": {name: "synthetic-scorer" for name in runner.SCORING_FILES},
            "working_tree": "",
            "dirty": False,
            "diff_summary": "",
        },
    )
    monkeypatch.setattr(runner, "capture_code_changes", lambda root, run, snapshot: None)

    def fake_command(argv, root, run, name, commands, **kwargs):
        code = state["test_exit" if name == "tests" else "benchmark_exit"]
        commands.append({"argv": argv, "exit_code": code})
        if name == "tests":
            (run / "tests.xml").write_text(f"<testsuite>{state['test_body']}</testsuite>")
        elif code == 0:
            request = json.loads((run / "request.json").read_text())
            scores(expected, state["score"]).to_csv(run / "scores.csv", index=False)
            write_json(
                run / "capture.json",
                {
                    "run_id": run.name,
                    "scores_sha256": digest(run / "scores.csv"),
                    "config_sha256": json_digest(request["config"]),
                    "judgements_sha256": "synthetic-judgements",
                },
            )
        elif code == 2:
            write_json(
                run / "worker_error.json",
                {"status": "NOT RUN", "reason": state.get("worker_error", "fixture dependency")},
            )
        return code

    monkeypatch.setattr(runner, "command", fake_command)
    return root, args, state


def test_quick_pass_does_not_claim_a_real_benchmark(run_fixture):
    root, args, _ = run_fixture
    code, summary = runner.run_validation(args, root)
    manifest = verify_run(Path(summary["run_dir"]), summary["run_id"])
    assert code == 0 and manifest["status"] == "PASS"
    checks = json.loads((Path(summary["run_dir"]) / "checks.json").read_text())
    real = next(c for c in checks if c["name"] == "real_benchmark")
    assert real["status"] == "NOT RUN" and real["effect"] == "NOT EVALUATED"
    assert "fixture-private-ad" not in Path(summary["public_report"]).read_text()
    assert manifest["evaluator_sha256"] == json_digest(manifest["code"]["files"])


@pytest.mark.parametrize("profile", ["quick", "hook"])
def test_prompt_edit_gets_an_explicit_answer_quality_limit(run_fixture, monkeypatch, profile):
    root, args, _ = run_fixture
    args.profile = profile
    prompt_name = "src/job_market_intelligence/rag/prompts.py"
    assert hook.relevant_path(root, prompt_name)
    prompt = root / prompt_name
    prompt.parent.mkdir(parents=True)
    prompt.write_text('ANSWER_SYSTEM_PROMPT = "Revised prompt"\n')
    snapshot = runner.code_snapshot(root)
    snapshot["files"][prompt_name] = digest(prompt)
    snapshot.update(dirty=True, fingerprint=json_digest(snapshot["files"]))
    monkeypatch.setattr(runner, "code_snapshot", lambda root: snapshot)
    code, result = runner.run_validation(args, root)
    assert code == 0
    report = Path(result["public_report"])
    public = json.loads(report.with_name("summary.json").read_text())
    answer = next(c for c in public["checks"] if c["name"] == "answer_quality")
    assert answer["status"] == "NOT RUN" and answer["effect"] == "NOT EVALUATED"
    assert answer["required"] is False
    assert "Answer quality has not been measured end to end" in report.read_text()
    assert answer["reason"] in report.read_text()
    actual = verify_run(Path(result["run_dir"]))
    targets = actual["commands"][0]["argv"][3:-3]
    assert public["test_targets"] == actual["test_targets"] == targets
    assert all(f"`{target}`" in report.read_text() for target in targets)


@pytest.mark.parametrize("rag_target", ["tests/test_rag_routing.py", "tests/test_rag_stats.py"])
def test_hook_profile_fails_when_a_selected_rag_test_fails(run_fixture, monkeypatch, rag_target):
    root, args, state = run_fixture
    args.profile = "hook"
    command = runner.command

    def fail_rag_test_if_selected(argv, *args, **kwargs):
        if rag_target in argv:
            state.update(
                test_exit=1, test_body='<testcase name="rag_regression"><failure/></testcase>'
            )
        return command(argv, *args, **kwargs)

    monkeypatch.setattr(runner, "command", fail_rag_test_if_selected)
    code, result = runner.run_validation(args, root)
    assert code == 1 and result["status"] == "FAIL"
    checks = json.loads((Path(result["run_dir"]) / "checks.json").read_text())
    assert next(c for c in checks if c["name"] == "tests")["counts"]["failed"] == 1


@pytest.mark.parametrize("exception", [FileNotFoundError, ValueError])
def test_public_reasons_exclude_private_exception_details(run_fixture, monkeypatch, exception):
    root, args, _ = run_fixture
    private_message = f"{root}/private/fixture-private-ad.csv | raw diagnostic\nnext line"

    def fail_query_loading(*args):
        raise exception(private_message)

    monkeypatch.setattr(runner, "expected_queries", fail_query_loading)
    code, result = runner.run_validation(args, root)
    assert code == (2 if exception is FileNotFoundError else 1)
    checks = json.loads((Path(result["run_dir"]) / "checks.json").read_text())
    diagnostic = next(c for c in checks if "private_detail" in c)
    assert private_message in diagnostic["private_detail"]
    report = Path(result["public_report"])
    public = json.loads(report.with_name("summary.json").read_text())
    exported = next(c for c in public["checks"] if c["name"] == diagnostic["name"])
    assert exported["reason"] == diagnostic["reason"]
    assert exported["reason"] in report.read_text()
    assert "private_detail" not in exported
    assert "fixture-private-ad" not in report.read_text() + json.dumps(public)


def test_missing_benchmark_dependency_has_a_public_reason_and_private_detail(run_fixture):
    root, args, state = run_fixture
    args.mode = "baseline"
    state.update(benchmark_exit=2, worker_error="/private/fixture-private-ad/model unavailable")
    code, result = runner.run_validation(args, root)
    assert code == 2
    report = Path(result["public_report"])
    public = json.loads(report.with_name("summary.json").read_text())
    benchmark = next(c for c in public["checks"] if c["name"] == "benchmark_capture")
    assert benchmark["status"] == "NOT RUN"
    assert benchmark["reason"] == "benchmark dependency unavailable"
    assert benchmark["reason"] in report.read_text()
    assert "fixture-private-ad" not in report.read_text() + json.dumps(public)


def test_public_evidence_identifies_dirty_source_and_policy(run_fixture, monkeypatch):
    root, args, _ = run_fixture
    snapshot = runner.code_snapshot(root) | {"dirty": True, "working_tree": "private detail"}
    monkeypatch.setattr(runner, "code_snapshot", lambda root: snapshot)
    _, summary = runner.run_validation(args, root)
    manifest = verify_run(Path(summary["run_dir"]))
    public_report = Path(summary["public_report"])
    public = json.loads(public_report.with_name("summary.json").read_text())
    assert public["code_dirty"] is True
    assert public["code_commit"] == snapshot["commit"]
    assert public["code_fingerprint"] == snapshot["fingerprint"]
    assert public["policy_sha256"] == digest(args.policy)
    assert public["evaluator_sha256"] == manifest["evaluator_sha256"]
    text = public_report.read_text()
    assert "Working tree: contains uncommitted changes" in text
    assert public["policy_sha256"] in text and public["code_fingerprint"] in text
    assert "private detail" not in text + json.dumps(public)


def test_full_requires_an_explicit_baseline(run_fixture):
    root, args, _ = run_fixture
    args.mode = "full"
    code, summary = runner.run_validation(args, root)
    assert code == 2 and summary["status"] == "NOT RUN"
    assert not (Path(summary["run_dir"]) / "scores.csv").exists()
    public = json.loads(Path(summary["public_report"]).with_name("summary.json").read_text())
    baseline = next(c for c in public["checks"] if c["name"] == "baseline")
    report = Path(summary["public_report"]).read_text()
    assert "| Reason |" in report and baseline["reason"] in report


def test_capture_then_compare_and_reject_changed_inputs(run_fixture):
    root, args, state = run_fixture
    args.mode = "baseline"
    code, baseline = runner.run_validation(args, root)
    assert code == 0
    args.mode, args.baseline = "full", Path(baseline["run_dir"])
    state["score"] = 0.49
    code, candidate = runner.run_validation(args, root)
    assert code == 1 and candidate["status"] == "FAIL"
    state["score"] = 0.6
    code, improved = runner.run_validation(args, root)
    assert code == 0 and improved["status"] == "PASS"
    public = json.loads(Path(improved["public_report"]).with_name("summary.json").read_text())
    assert all(c["effect"] == "IMPROVEMENT" for c in public["comparisons"])
    assert next(c for c in public["checks"] if c["name"] == "answer_quality")["status"] == "NOT RUN"
    assert "Answer quality has not been measured end to end" in Path(improved["report"]).read_text()
    args.document_labels.write_text("changed input")
    code, changed = runner.run_validation(args, root)
    assert code == 1
    assert not (Path(changed["run_dir"]) / "scores.csv").exists()


def test_stale_run_and_tampering_are_rejected(run_fixture, tmp_path):
    root, args, _ = run_fixture
    _, summary = runner.run_validation(args, root)
    run = Path(summary["run_dir"])
    with pytest.raises(ValueError, match="identity"):
        verify_run(run, "old-run-id")
    copied = tmp_path / "copied-run"
    shutil.copytree(run, copied)
    with pytest.raises(ValueError, match="identity"):
        verify_run(copied)
    (run / "checks.json").write_text("[]")
    with pytest.raises(ValueError, match="hash"):
        verify_run(run)


def test_legacy_csv_cannot_be_blessed_as_a_baseline(tmp_path):
    (tmp_path / "scores.csv").write_text("query_id,ndcg@10\nx,0.5\n")
    with pytest.raises(ValueError, match="manifest"):
        load_baseline(tmp_path, {}, "policy", "scorer")


def test_worker_refuses_to_overwrite_a_completed_run(run_fixture):
    from job_market_intelligence.retrieval.validation_benchmark import worker

    root, args, _ = run_fixture
    args.mode = "baseline"
    _, summary = runner.run_validation(args, root)
    run = Path(summary["run_dir"])
    original_hash = digest(run / "scores.csv")
    with pytest.raises(ValueError, match="overwrite"):
        worker(run)
    assert digest(run / "scores.csv") == original_hash


@pytest.mark.parametrize("scoring_file", runner.SCORING_FILES)
def test_changed_scorer_cannot_be_reported_as_retrieval_improvement(
    run_fixture, monkeypatch, scoring_file
):
    root, args, state = run_fixture
    args.mode = "baseline"
    code, baseline = runner.run_validation(args, root)
    assert code == 0
    args.mode, args.baseline = "full", Path(baseline["run_dir"])
    state["score"] = 0.8
    snapshot = runner.code_snapshot(root)
    snapshot["files"][scoring_file] = "different-scorer"
    monkeypatch.setattr(runner, "code_snapshot", lambda root: snapshot)
    code, candidate = runner.run_validation(args, root)
    assert code == 1
    assert not (Path(candidate["run_dir"]) / "scores.csv").exists()


def test_sealing_failure_does_not_publish_a_pass(run_fixture, monkeypatch):
    root, args, _ = run_fixture

    def fail_sealing(*args):
        raise OSError("injected sealing failure")

    monkeypatch.setattr(runner, "seal_run", fail_sealing)
    with pytest.raises(OSError, match="sealing"):
        runner.run_validation(args, root)
    assert not list((root / "reports/validation").rglob("summary.json"))


@pytest.mark.parametrize("fault", ["zero", "failure", "all_skipped"])
def test_runner_never_passes_invalid_test_execution(run_fixture, fault):
    root, args, state = run_fixture
    state["test_body"] = {
        "zero": "",
        "failure": "<testcase><failure/></testcase>",
        "all_skipped": "<testcase><skipped/></testcase>",
    }[fault]
    code, summary = runner.run_validation(args, root)
    assert code != 0 and summary["status"] != "PASS"


def test_privacy_leak_fails_the_validation_command(run_fixture):
    root, args, _ = run_fixture
    (root / "eval/leak.txt").write_text("fixture-private-ad")
    code, summary = runner.run_validation(args, root)
    assert code == 1 and summary["status"] == "FAIL"


@pytest.mark.parametrize("mutation", ["add", "change", "delete"])
def test_publication_changes_invalidate_current_privacy_evidence(run_fixture, mutation):
    root, args, _ = run_fixture
    folder = root / "reports/retrieval"
    folder.mkdir(parents=True)
    path = folder / "example.txt"
    path.write_text("aggregate results")
    _, summary = runner.run_validation(args, root)
    manifest = verify_run(Path(summary["run_dir"]))
    verify_current_publication(root, manifest)
    if mutation == "add":
        (folder / "new.txt").write_text("fixture-private-ad")
    elif mutation == "change":
        path.write_text("fixture-private-ad")
    else:
        path.unlink()
    with pytest.raises(ValueError, match="published"):
        verify_current_publication(root, manifest)


def test_publication_mutation_during_scan_fails_the_run(run_fixture, monkeypatch):
    root, args, _ = run_fixture

    def mutate_after_scan(paths, jobs):
        (root / "eval/new-leak.txt").write_text("fixture-private-ad")
        return []

    monkeypatch.setattr(runner, "privacy_offenders", mutate_after_scan)
    code, _ = runner.run_validation(args, root)
    assert code == 1


def test_source_change_during_run_invalidates_results(run_fixture, monkeypatch):
    root, args, _ = run_fixture
    before = runner.code_snapshot(root)
    snapshots = iter([before, before | {"fingerprint": "changed"}])
    monkeypatch.setattr(runner, "code_snapshot", lambda root: next(snapshots))
    code, summary = runner.run_validation(args, root)
    assert code == 1
    checks = json.loads((Path(summary["run_dir"]) / "checks.json").read_text())
    assert next(c for c in checks if c["name"] == "tests")["status"] == "PASS"
    assert next(c for c in checks if c["name"] == "source_stability")["status"] == "FAIL"


def test_missing_private_corpus_is_visible_but_optional_for_quick(run_fixture):
    root, args, _ = run_fixture
    args.jobs.unlink()
    code, summary = runner.run_validation(args, root)
    checks = json.loads((Path(summary["run_dir"]) / "checks.json").read_text())
    assert code == 0
    privacy = next(c for c in checks if c["name"] == "private_corpus_privacy")
    assert privacy["status"] == "NOT RUN"
    assert privacy["reason"] in Path(summary["public_report"]).read_text()


def test_hook_filters_paths_and_handles_missing_python(tmp_path):
    event = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Edit",
        "tool_input": {"file_path": str(tmp_path / "README.md")},
    }
    assert hook.handle(event, tmp_path) is None
    assert not hook.relevant_path(tmp_path, "../outside.py")
    event["tool_input"]["file_path"] = str(tmp_path / "eval/validation_policy.json")
    result = hook.handle(event, tmp_path)
    assert "NOT RUN" in result["hookSpecificOutput"]["additionalContext"]


@pytest.mark.parametrize("timeout", [False, True])
def test_hook_runs_only_lightweight_checks(tmp_path, monkeypatch, timeout):
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()

    def fake_run(argv, **kwargs):
        assert "quick" in argv and "full" not in argv and "baseline" not in argv
        if timeout:
            raise subprocess.TimeoutExpired(argv, 40)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"status": "PASS", "run_id": "fixture", "report": "/private/report.md"}
            ),
        )

    monkeypatch.setattr(hook.subprocess, "run", fake_run)
    result = hook.handle(
        {
            "hook_event_name": "PostToolUse",
            "tool_name": "Write",
            "tool_input": {"file_path": "eval/validation_policy.json"},
        },
        tmp_path,
    )
    message = result["hookSpecificOutput"]["additionalContext"]
    assert "NOT RUN" in message
    if not timeout:
        assert "PASS for the hook test selection" in message
        assert "Only the selected tests were checked" in message
        assert "Answer quality: NOT RUN" in message


def test_capture_runs_real_scoring_on_synthetic_inputs(tmp_path, expected, monkeypatch):
    import duckdb

    from job_market_intelligence.retrieval import validation_benchmark as benchmark
    from job_market_intelligence.retrieval.config import RetrievalConfig

    # A tiny local encoder exercises fresh chunking, retrieval, judgement resolution
    # and scoring without downloading weights. This is a synthetic acceptance test.
    class FixtureEncoder:
        def __init__(self, *args, **kwargs):
            assert kwargs.get("cache_dir") is None
            self.model = self

        def encode(self, texts, **kwargs):
            vectors = []
            for text in texts:
                if kwargs.get("is_query"):
                    vectors.append([1.0, 0.0] if "Python" in text else [0.0, 1.0])
                elif "Python" in text:
                    vectors.append([0.6, -0.8])
                elif "SQL" in text:
                    vectors.append([-0.8, 0.6])
                else:
                    vectors.append([2**-0.5, 2**-0.5])
            return np.asarray(vectors, dtype=np.float32)

    monkeypatch.setattr(benchmark, "ChunkEncoder", FixtureEncoder)
    jobs = tmp_path / "jobs.csv"
    pd.DataFrame(
        [
            {
                "job_id": "fictional-a",
                "title": "Analyst",
                "city": "Melbourne",
                "description": "Use Python to explain research results and build analyses. " * 12,
            },
            {
                "job_id": "fictional-b",
                "title": "Engineer",
                "city": "Melbourne",
                "description": "Use SQL to build reliable data pipelines and production systems. "
                * 12,
            },
            {
                "job_id": "fictional-unjudged",
                "title": "Generalist",
                "city": "Sydney",
                "description": "General skills and experience are useful in this fictional role.",
            },
        ]
    ).to_csv(jobs, index=False)
    labels = tmp_path / "labels.csv"
    labels.write_text("job_id\nfictional-a\nfictional-b\n")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "\n".join(
            json.dumps(
                {
                    "query_id": key,
                    "query_class": group,
                    "query": "Python roles" if key.endswith("0") else "SQL roles",
                    "judged_pool": "labelled" if key.endswith("0") else "all",
                    "relevance": {
                        "type": "metadata_match",
                        "column": "job_id",
                        "values": ["fictional-a" if key.endswith("0") else "fictional-b"],
                    },
                }
            )
            for key, group in expected.items()
        )
    )
    database = tmp_path / "data.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "CREATE TABLE skill_mentions (job_id VARCHAR, skill VARCHAR, "
            "category VARCHAR, context VARCHAR)"
        )
    inputs = {"jobs": jobs, "document_labels": labels, "queries": queries, "database": database}
    request = {
        "config": asdict(RetrievalConfig(mode="dense", chunk_tokens=32)),
        "run_id": "synthetic",
        "input_paths": {k: str(v) for k, v in inputs.items()},
        "input_hashes": {k: digest(v) for k, v in inputs.items()},
        "expected_queries": expected,
    }
    run = tmp_path / "capture"
    run.mkdir()
    benchmark.capture(request, run)
    frame = pd.read_csv(run / "scores.csv")
    assert len(frame) == len(expected)
    for row in frame.to_dict(orient="records"):
        expected_ndcg = 1.0 if row["query_id"].endswith("0") else 1 / np.log2(3)
        assert row["ndcg@10"] == pytest.approx(expected_ndcg)
    chunks = pd.read_parquet(run / "chunks.parquet")
    assert chunks.groupby("job_id").size().max() > 1
    capture = json.loads((run / "capture.json").read_text())
    assert capture["scores_sha256"] == digest(run / "scores.csv")
    assert (run / "embeddings.npy").is_file() and (run / "chunks.parquet").is_file()


def test_no_checks_cannot_pass():
    assert overall_status([]) == "NOT RUN"
