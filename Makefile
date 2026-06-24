.PHONY: install pipeline annotations benchmark public-release test lint app public-app purge-private check

install:
	python3 -m venv .venv
	.venv/bin/python -m pip install -e ".[dev]"

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
