"""Checks on the deployment artefacts that do not require a Docker daemon.

The daemon-dependent acceptance checks live in `make docker-verify`; these are
the ones that must fail fast in the ordinary test run, because they guard
against the failure mode that matters: `.dockerignore` drifting away from the
privacy rules and quietly letting licensed text into an image layer.
"""

import re
from pathlib import Path

import pytest

from job_market_intelligence.privacy import (
    BUILD_EXCLUDED_PATHS,
    KEEP_PATHS,
    PRIVATE_PATHS,
    dockerignore_is_current,
    render_dockerignore,
)

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yml"


def test_dockerignore_matches_the_privacy_rules():
    """The generated file and the rules must not drift; run `make dockerignore`."""

    assert dockerignore_is_current(ROOT), (
        ".dockerignore is stale. Regenerate it with `make dockerignore`."
    )


def test_every_private_path_is_excluded_from_the_build_context():
    body = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for path in PRIVATE_PATHS:
        assert path in body, f"{path} is not excluded from the image build context"


def test_the_synthetic_corpus_survives_the_exclusions():
    """Excluding all of data/ would make the image unable to run the pipeline."""

    assert "!data/synthetic/retrieval_corpus.csv" in render_dockerignore()
    assert any(rule.startswith("!") for rule in KEEP_PATHS)


def test_build_noise_is_excluded_but_is_not_confused_with_private_data():
    assert ".venv/" in BUILD_EXCLUDED_PATHS
    # Build noise and licensed data are separate lists so that loosening one
    # cannot silently loosen the other.
    assert not set(PRIVATE_PATHS) & set(BUILD_EXCLUDED_PATHS)


@pytest.mark.skipif(not DOCKERFILE.exists(), reason="no Dockerfile")
class TestDockerfile:
    @staticmethod
    def body() -> str:
        return DOCKERFILE.read_text(encoding="utf-8")

    def test_installs_cpu_only_torch_before_anything_else(self):
        body = self.body()
        assert "download.pytorch.org/whl/cpu" in body
        # Ordering matters: a transitive dependency resolved first would pull
        # the CUDA build and add over a gigabyte of unusable driver libraries.
        assert body.index("whl/cpu") < body.index("pip install -r requirements.txt")

    def test_pins_model_revisions_to_commits(self):
        body = self.body()
        for argument in ("EMBED_REVISION", "RERANK_REVISION"):
            match = re.search(rf"ARG {argument}=([0-9a-f]{{40}})", body)
            assert match, f"{argument} must default to a 40-character commit sha"

    def test_baked_revisions_match_the_ones_the_code_pins(self):
        from job_market_intelligence.retrieval.embedding import MODEL_REVISIONS
        from job_market_intelligence.retrieval.rerank import RERANK_REVISIONS

        body = self.body()
        embed = re.search(r"ARG EMBED_REVISION=([0-9a-f]{40})", body).group(1)
        rerank = re.search(r"ARG RERANK_REVISION=([0-9a-f]{40})", body).group(1)
        assert embed == MODEL_REVISIONS["BAAI/bge-small-en-v1.5"]
        assert rerank == RERANK_REVISIONS["cross-encoder/ms-marco-MiniLM-L-6-v2"]

    def test_runs_offline_so_a_missing_weight_fails_loudly(self):
        assert "HF_HUB_OFFLINE=1" in self.body()

    def test_runs_as_a_non_root_user(self):
        body = self.body()
        assert "useradd" in body and "USER app" in body
        # Compare instruction positions, not substrings: "CMD" also appears in
        # the header comment.
        instructions = [line.split()[0] for line in body.splitlines() if line[:1].isupper()]
        assert instructions.index("USER") < instructions.index("CMD")

    def test_base_image_is_pinned_rather_than_latest(self):
        stages = set()
        for line in self.body().splitlines():
            if not line.startswith("FROM "):
                continue
            parts = line.split()
            image = parts[1]
            if "AS" in parts:
                stages.add(parts[parts.index("AS") + 1])
            # A FROM referencing an earlier stage carries no tag by design.
            if image in stages:
                continue
            assert ":latest" not in line
            assert ":" in image, f"unpinned base image: {image}"

    def test_dependencies_are_installed_before_source_is_copied(self):
        body = self.body()
        # Otherwise every source edit reinstalls torch.
        assert body.index("COPY pyproject.toml ./") < body.index("COPY --chown=app:app src/")

    def test_no_secret_is_baked_into_the_image(self):
        body = self.body()
        for secret in ("ADZUNA_APP_KEY", "JSEARCH_API_KEY", "LLM_API_KEY"):
            assert secret not in body, f"{secret} must be injected at run time only"


@pytest.mark.skipif(not COMPOSE.exists(), reason="no docker-compose.yml")
class TestCompose:
    @staticmethod
    def body() -> str:
        return COMPOSE.read_text(encoding="utf-8")

    def test_api_is_published_to_loopback_only(self):
        body = self.body()
        assert '"127.0.0.1:8000:8000"' in body
        assert '"8000:8000"' not in body.replace('"127.0.0.1:8000:8000"', "")

    def test_cpu_limit_is_set_so_latency_figures_are_reproducible(self):
        body = self.body()
        assert "cpus:" in body and "memory:" in body

    def test_index_build_runs_before_the_api_serves(self):
        assert "service_completed_successfully" in self.body()

    def test_secrets_arrive_through_env_file_not_the_image(self):
        assert "env_file" in self.body()
