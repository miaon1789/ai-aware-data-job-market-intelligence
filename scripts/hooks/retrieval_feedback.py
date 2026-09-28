#!/usr/bin/env python3
"""Claude Code PostToolUse adapter. JSON in, concise feedback out, no model calls."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path


def relevant_path(root: Path, value: str) -> bool:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return False
    return relative.startswith(
        (
            "src/job_market_intelligence/retrieval/",
            "src/job_market_intelligence/rag/",
            "eval/golden/",
            "tests/test_retrieval",
            "tests/test_golden",
            "tests/test_validation",
            "tests/test_chunking",
            "tests/test_rag_",
            "scripts/hooks/",
            ".claude/skills/validate-retrieval/",
        )
    ) or relative in {
        "eval/validation_policy.json",
        "scripts/validate_retrieval.py",
        "src/job_market_intelligence/privacy.py",
        "scripts/run_retrieval_ablation.py",
        ".claude/agents/retrieval-reviewer.md",
        ".claude/settings.json",
        ".gitignore",
        ".dockerignore",
        "pyproject.toml",
    }


def feedback(message: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": message}}


def handle(event: dict, root: Path) -> dict | None:
    if event.get("hook_event_name") != "PostToolUse" or event.get("tool_name") not in {
        "Edit",
        "Write",
    }:
        return None
    path = event.get("tool_input", {}).get("file_path")
    if not isinstance(path, str) or not relevant_path(root, path):
        return None
    python = root / ".venv/bin/python"
    if not python.is_file():
        return feedback("Retrieval hook: NOT RUN. Project .venv/bin/python is unavailable.")
    state = root / "data/private/validation"
    state.mkdir(parents=True, mode=0o700, exist_ok=True)
    with (state / ".hook.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return feedback("Retrieval hook: NOT RUN for this edit. Another hook check is running.")
        try:
            result = subprocess.run(
                [
                    str(python),
                    str(root / "scripts/validate_retrieval.py"),
                    "quick",
                    "--profile",
                    "hook",
                    "--test-timeout",
                    "15",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=40,
                check=False,
            )
            summary = json.loads(result.stdout.strip().splitlines()[-1])
            status = summary["status"]
            expected_exit = {"PASS": 0, "FAIL": 1, "NOT RUN": 2}[status]
            if result.returncode != expected_exit:
                raise ValueError("runner exit code disagrees with its summary")
            return feedback(
                f"Retrieval hook ({summary['run_id']}): {status} for the hook test selection. "
                "Only the selected tests were checked. "
                "Answer quality: NOT RUN (not measured end to end). "
                f"Real benchmark: NOT RUN. Evidence: {summary['report']}. "
                "Use /validate-retrieval quick for the full lightweight check."
            )
        except (subprocess.TimeoutExpired, OSError, ValueError, KeyError, IndexError):
            return feedback(
                "Retrieval hook: NOT RUN. The runner timed out or returned invalid evidence. "
                "Run /validate-retrieval quick. Incomplete run directories are not valid evidence."
            )


def main() -> int:
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2]))
    try:
        event = json.load(sys.stdin)
        result = handle(event, root)
    except (ValueError, OSError, AttributeError):
        result = feedback("Retrieval hook: NOT RUN. Hook input or local state was unavailable.")
    if result:
        print(json.dumps(result))
    return 0  # Feedback after an edit cannot undo that edit.


if __name__ == "__main__":
    raise SystemExit(main())
