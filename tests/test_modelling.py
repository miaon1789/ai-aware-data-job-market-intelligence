import pandas as pd

from job_market_intelligence.modelling import predict_role, train_and_evaluate
from scripts.generate_synthetic_data import generate_records


def test_grouped_model_training_runs_and_predicts_known_role():
    frame = pd.DataFrame(generate_records()[:-3])
    result = train_and_evaluate(frame)
    assert result.metrics["folds"] >= 2
    assert result.metrics["samples"] == len(frame)
    prediction, score = predict_role(
        result.model,
        "Graduate Data Engineer",
        "Build SQL data pipelines with Python, dbt, Airflow and data modelling.",
    )
    assert prediction == "Data Engineer"
    assert 0.0 <= score <= 1.0
