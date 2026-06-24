import duckdb
import pandas as pd

from scripts.generate_synthetic_data import generate_csv
from scripts.run_pipeline import run


def test_unlabelled_input_uses_separate_training_data_and_redacts_contacts(tmp_path):
    training_path = generate_csv(tmp_path / "training.csv")
    input_frame = pd.read_csv(training_path, keep_default_na=False)
    input_frame["role_label"] = ""
    input_frame.loc[10, "description"] += " Contact person@example.com or 0412 345 678."
    input_path = tmp_path / "unlabelled.csv"
    input_frame.to_csv(input_path, index=False)

    output_dir = tmp_path / "processed"
    summary = run(input_path, training_path, output_dir, tmp_path / "reports")
    assert summary["clean_rows"] == 54
    with duckdb.connect(str(output_dir / "job_market.duckdb"), read_only=True) as connection:
        missing_roles = connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE analysis_role IS NULL"
        ).fetchone()[0]
        stored_description = connection.execute(
            "SELECT description FROM jobs WHERE job_id = 'SYN-0011'"
        ).fetchone()[0]
    assert missing_roles == 0
    assert "person@example.com" not in stored_description
    assert "0412 345 678" not in stored_description
