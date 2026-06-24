import os

import pytest

from scripts.fetch_adzuna import ROOT, load_env_file, safe_query_slug, validate_private_output


def test_env_loader_reads_only_expected_secrets(tmp_path, monkeypatch):
    monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
    monkeypatch.delenv("ADZUNA_APP_KEY", raising=False)
    monkeypatch.delenv("UNRELATED", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ADZUNA_APP_ID=test-id\nADZUNA_APP_KEY='test-key'\nUNRELATED=do-not-load\n"
    )
    load_env_file(env_file)
    assert os.environ["ADZUNA_APP_ID"] == "test-id"
    assert os.environ["ADZUNA_APP_KEY"] == "test-key"
    assert "UNRELATED" not in os.environ


def test_row_level_output_is_forced_under_private_directory():
    allowed = validate_private_output(ROOT / "data/private/adzuna/jobs.csv")
    assert allowed.name == "jobs.csv"
    with pytest.raises(ValueError, match="data/private"):
        validate_private_output(ROOT / "data/public/jobs.csv")


def test_query_slug_cannot_escape_raw_response_directory():
    assert safe_query_slug("../../Data Analyst") == "data-analyst"
