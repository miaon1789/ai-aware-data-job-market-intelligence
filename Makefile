.PHONY: install install-retrieval pipeline annotations benchmark public-release test lint app \
	public-app purge-private check retrieval-corpus chunks index golden ablation cost \
	retrieval synthetic-corpus api mcp routing dockerignore docker-build \
	docker-build-multi docker-run docker-stop docker-verify docker-offline-check

install:
	python3 -m venv .venv
	.venv/bin/python -m pip install -e ".[dev]"

install-retrieval:
	.venv/bin/python -m pip install -e ".[dev,retrieval]"

pipeline:
	.venv/bin/python scripts/run_pipeline.py

annotations:
	.venv/bin/python scripts/prepare_annotations.py

benchmark:
	.venv/bin/python scripts/fetch_skillspan.py
	.venv/bin/python scripts/evaluate_skillspan.py

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check .

app:
	.venv/bin/streamlit run app/streamlit_app.py

public-release:
	.venv/bin/python scripts/build_public_release.py

public-app:
	.venv/bin/streamlit run app/public_dashboard.py

purge-private:
	.venv/bin/python scripts/purge_private_data.py

check: lint test

# --- Part 2: retrieval, evaluation and services -------------------------------
# Rebuild the private retrieval corpus from the collectors. Needs Adzuna
# credentials in .env; the ATS boards are public.
retrieval-corpus:
	.venv/bin/python scripts/fetch_ats.py
	.venv/bin/python scripts/merge_job_ads.py \
		--input data/private/ats/job_ads.csv \
		--input data/private/adzuna/job_ads_expanded.csv \
		--output data/private/merged/job_ads_retrieval.csv
	.venv/bin/python scripts/run_pipeline.py \
		--input data/private/merged/job_ads_retrieval.csv \
		--output-dir data/private/retrieval \
		--reports-dir data/private/retrieval/reports \
		--max-age-days 3650

chunks:
	.venv/bin/python scripts/build_chunks.py

index:
	.venv/bin/python scripts/build_retrieval_index.py

golden:
	.venv/bin/python scripts/build_golden_set.py

ablation:
	.venv/bin/python scripts/run_retrieval_ablation.py

cost:
	.venv/bin/python scripts/measure_retrieval_cost.py

routing:
	.venv/bin/python scripts/evaluate_routing.py

# Everything from an existing corpus through to the published evaluation tables.
retrieval: chunks index golden ablation cost routing

# Fictional full-text advertisements, for running the pipeline without the
# licensed corpus.
synthetic-corpus:
	.venv/bin/python scripts/generate_synthetic_corpus.py

# Local only: the index is built from licensed advertisement text.
api:
	.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

mcp:
	.venv/bin/python mcp_server/server.py

# --- Deployment ---------------------------------------------------------------
IMAGE ?= job-market-intelligence:local

# .dockerignore is generated, never hand-edited: it and the privacy tests read
# the same rules, so they cannot drift apart.
dockerignore:
	.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); \
from pathlib import Path; \
from job_market_intelligence.privacy import render_dockerignore; \
Path('.dockerignore').write_text(render_dockerignore(), encoding='utf-8'); \
print('wrote .dockerignore')"

docker-build: dockerignore
	docker build -t $(IMAGE) .
	@docker image inspect $(IMAGE) --format 'image size: {{.Size}} bytes'

# Publish for both architectures: a Mac builds arm64 by default, which fails
# with "exec format error" on an x86 host.
docker-build-multi: dockerignore
	docker buildx build --platform linux/amd64,linux/arm64 -t $(IMAGE) .

docker-run:
	docker compose up --build

docker-stop:
	docker compose down

# Privacy acceptance: assert the built image contains no licensed data. This is
# the check that turns the repository's privacy claim into an enforced property
# of the artefact rather than a promise in a README.
docker-verify:
	@echo "checking image for licensed data..."
	@docker run --rm --entrypoint sh $(IMAGE) -c '\
	  ! test -e /app/data/private && \
	  ! test -e /app/data/raw && \
	  ! test -e /app/data/processed && \
	  ! test -e /app/.env && \
	  test -f /app/data/synthetic/retrieval_corpus.csv' \
	  && echo "  no private data in image; synthetic corpus present"
	@test "$$(docker run --rm --entrypoint id $(IMAGE) -u)" != "0" \
	  && echo "  runs as non-root"
	@docker run --rm --network none --entrypoint python $(IMAGE) -c '\
	from job_market_intelligence.retrieval.embedding import ChunkEncoder; \
	from job_market_intelligence.retrieval.rerank import CrossEncoderReranker; \
	ChunkEncoder().encode(["offline smoke test"], is_query=True); \
	CrossEncoderReranker().rerank("q", [("c", "text")], top_k=1); \
	print("  both models load with no network")'
	@docker run --rm --network none --entrypoint python $(IMAGE) -c '\
	import duckdb; duckdb.connect().execute("INSTALL fts; LOAD fts;"); \
	print("  duckdb fts extension loads with no network")'

# Full offline acceptance: build the whole pipeline from the synthetic corpus
# inside the container with no network, then run a routed query. This is the
# check that found the DuckDB extension download; nothing static would have.
docker-offline-check:
	@rm -rf .docker-offline && mkdir -p .docker-offline
	@chmod 777 .docker-offline
	@docker run --rm --network none -v $(PWD)/.docker-offline:/app/data/private \
	  --entrypoint sh $(IMAGE) -c '\
	  set -e; \
	  python scripts/run_pipeline.py --input data/synthetic/retrieval_corpus.csv \
	    --output-dir data/private/retrieval --reports-dir data/private/retrieval/reports \
	    --max-age-days 3650 >/dev/null 2>&1; \
	  python scripts/build_chunks.py --jobs data/private/retrieval/jobs_clean.csv \
	    --chunk-tokens 512 >/dev/null 2>&1; \
	  python scripts/build_retrieval_index.py \
	    --embedding-model BAAI/bge-small-en-v1.5 >/dev/null 2>&1; \
	  echo "  built chunks, BM25 and dense indexes offline"'
	@rm -rf .docker-offline
