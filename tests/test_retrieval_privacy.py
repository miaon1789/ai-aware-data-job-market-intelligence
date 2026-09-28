"""Guards on what the retrieval work is allowed to publish.

The repository's data posture is that licensed advertisement text and the ids
that resolve to it stay private, and only aggregates and hashes are committed.
These tests fail if a future change starts writing either into a tracked file.
They skip rather than fail when the private corpus is absent, so a clone
without the data still runs a green suite.
"""

import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "data/private/retrieval/jobs_clean.csv"
PUBLISHED_DIRECTORIES = (ROOT / "eval", ROOT / "reports/retrieval", ROOT / "reports/validation")
MUST_STAY_PRIVATE = (
    "data/private/retrieval/golden/resolved.jsonl",
    "data/private/retrieval/chunks",
    "data/private/retrieval/index",
    "data/private/validation",
)


def published_files() -> list[Path]:
    return [
        path
        for directory in PUBLISHED_DIRECTORIES
        for path in directory.rglob("*")
        if path.is_file()
    ]


@pytest.mark.skipif(not JOBS.exists(), reason="private corpus is not present")
def test_published_artefacts_contain_no_advertisement_ids():
    job_ids = set(pd.read_csv(JOBS, dtype=str, keep_default_na=False)["job_id"])
    assert job_ids, "corpus should not be empty"

    offenders: list[str] = []
    for path in published_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for job_id in job_ids:
            if re.search(rf"(?<![\w-]){re.escape(job_id)}(?![\w-])", text):
                offenders.append(f"{path.relative_to(ROOT)} leaks {job_id}")
                break
    assert not offenders, "published artefacts leak advertisement ids: " + "; ".join(offenders)


@pytest.mark.skipif(not JOBS.exists(), reason="private corpus is not present")
def test_published_artefacts_contain_no_advertisement_text():
    jobs = pd.read_csv(JOBS, dtype=str, keep_default_na=False)
    # Normalize whitespace and check a fragment from every advertisement.
    fragments = [
        " ".join(str(description).split()[:8])
        for description in jobs["description"]
        if len(str(description).split()) >= 8
    ]
    published = {
        path: " ".join(path.read_text(encoding="utf-8", errors="replace").split())
        for path in published_files()
    }
    for fragment in fragments:
        for path, text in published.items():
            assert fragment not in text, f"{path.relative_to(ROOT)} contains advertisement text"


@pytest.mark.parametrize("relative_path", MUST_STAY_PRIVATE)
def test_private_retrieval_artefacts_are_git_ignored(relative_path):
    result = subprocess.run(
        ["git", "check-ignore", "-q", relative_path], cwd=ROOT, capture_output=True
    )
    assert result.returncode == 0, f"{relative_path} is not git-ignored"


def test_the_published_relevance_file_is_tracked_not_ignored():
    result = subprocess.run(
        ["git", "check-ignore", "-q", "eval/golden/relevance_hashed.jsonl"],
        cwd=ROOT,
        capture_output=True,
    )
    assert result.returncode != 0, "the published relevance file should be committable"
