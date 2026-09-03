# syntax=docker/dockerfile:1.7
#
# One image, two entrypoints. The FastAPI service and the MCP server share the
# retrieval package, both model weights and DuckDB; building them separately
# would duplicate ~400 MB of weights for no benefit. The entrypoint is chosen
# by CMD (see docker-compose.yml and the MCP client config in the README).
#
# What is deliberately NOT in this image: the advertisement corpus, the chunks,
# the embeddings and the BM25 index. Those are licensed text or derived from
# it, they are excluded by a .dockerignore generated from the same rules that
# drive the privacy tests, and `make docker-verify` asserts their absence in
# the built image. Indexes arrive at run time through a volume.

# ---------------------------------------------------------------- dependencies
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /build

# Only the dependency manifest, so that editing source code does not invalidate
# the layer that installs a few hundred megabytes of wheels.
COPY pyproject.toml ./
RUN python - <<'PY'
import tomllib
from pathlib import Path

# Single source of truth: dependencies are read from pyproject rather than kept
# in a second list that would drift from it.
project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
# The "service" extra, not the full dependency set: the container serves an API
# and builds indexes, it never renders the Streamlit dashboard.
required = project["optional-dependencies"]["service"]
Path("requirements.txt").write_text("\n".join(required) + "\n")
PY

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 # CPU-only torch, installed first so that no transitive dependency can pull
 # the CUDA build in behind it. The CUDA wheels add well over a gigabyte of
 # driver libraries that nothing in this project can use.
 && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && /opt/venv/bin/pip install -r requirements.txt \
 # Strip what only matters when building against torch, not when running it:
 # its bundled test suite and C++ headers, ~155 MB. torch/bin stays -- it holds
 # torch_shm_manager, which torch resolves at import time and fails without.
 && rm -rf /opt/venv/lib/python3.12/site-packages/torch/test \
           /opt/venv/lib/python3.12/site-packages/torch/include \
 && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} + \
 && find /opt/venv -name "*.pyc" -delete

# ------------------------------------------------------------- model weights
# Baked in rather than downloaded at run time, for two reasons: the container
# must work on an air-gapped host, and the nDCG figures in the README are only
# reproducible if the weights cannot change. Revisions are pinned to the same
# commits the retrieval package pins in code.
FROM builder AS models

ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ARG EMBED_REVISION=5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ARG RERANK_REVISION=233902d25c440f23af6f7d6e94d2946bac0bee0a
ENV HF_HOME=/opt/hf

RUN --mount=type=cache,target=/root/.cache/huggingface \
    EMBED_MODEL="$EMBED_MODEL" EMBED_REVISION="$EMBED_REVISION" \
    RERANK_MODEL="$RERANK_MODEL" RERANK_REVISION="$RERANK_REVISION" \
    /opt/venv/bin/python - <<'PY'
import os

from sentence_transformers import CrossEncoder, SentenceTransformer

SentenceTransformer(os.environ["EMBED_MODEL"], revision=os.environ["EMBED_REVISION"])
CrossEncoder(os.environ["RERANK_MODEL"], revision=os.environ["RERANK_REVISION"])
print("baked", os.environ["EMBED_MODEL"], "and", os.environ["RERANK_MODEL"])
PY

# DuckDB fetches its full-text-search extension from extensions.duckdb.org the
# first time a BM25 index is built. On a host with no network that is a hard
# failure at index-build time -- found by an offline acceptance run, and not by
# any amount of reading the Dockerfile. Bake it in like the model weights.
RUN mkdir -p /opt/duckdb \
 && HOME=/opt/duckdb /opt/venv/bin/python -c \
    "import duckdb; duckdb.connect().execute('INSTALL fts; LOAD fts;'); print('baked duckdb fts')" \
 && ls /opt/duckdb/.duckdb/extensions/*/*/

# ------------------------------------------------------------------- runtime
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="job-market-intelligence" \
      org.opencontainers.image.description="Hybrid retrieval and MCP tools over a private job-advertisement corpus" \
      org.opencontainers.image.licenses="MIT"

RUN useradd --system --uid 10001 --create-home app

COPY --from=builder /opt/venv /opt/venv
COPY --from=models  /opt/hf   /opt/hf
# DuckDB resolves extensions from $HOME/.duckdb, so this must land in the
# runtime user's home directory rather than root's.
COPY --from=models --chown=app:app /opt/duckdb/.duckdb /home/app/.duckdb

ENV PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/hf \
    # If a weight was not baked in, fail at startup instead of silently
    # fetching a different revision and quietly invalidating the evaluation.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:/app \
    RETRIEVAL_ROOT=/app/data/private/retrieval

WORKDIR /app

COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app src/        ./src/
COPY --chown=app:app api/        ./api/
COPY --chown=app:app mcp_server/ ./mcp_server/
COPY --chown=app:app scripts/    ./scripts/
COPY --chown=app:app config/     ./config/
COPY --chown=app:app eval/       ./eval/
COPY --chown=app:app data/synthetic/ ./data/synthetic/

RUN chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
