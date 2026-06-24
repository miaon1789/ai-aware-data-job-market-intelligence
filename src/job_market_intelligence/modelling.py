"""Leakage-aware role classification baseline."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline


@dataclass
class ModelResult:
    model: Pipeline
    metrics: dict[str, object]
    predictions: pd.DataFrame


def combine_text(frame: pd.DataFrame) -> pd.Series:
    return frame["title"].fillna("") + " [title] " + frame["description"].fillna("")


def build_classifier() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.98,
                    sublinear_tf=True,
                    max_features=12_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=42),
            ),
        ]
    )


def _evaluate_text(text: pd.Series, labels: pd.Series, groups: pd.Series, splits: int):
    estimator = build_classifier()
    cv = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=42)
    return cross_val_predict(estimator, text, labels, groups=groups, cv=cv, method="predict")


def train_and_evaluate(frame: pd.DataFrame) -> ModelResult:
    """Evaluate by company group, then fit the deployable model on all labels."""

    labelled = frame[frame["role_label"].notna() & frame["role_label"].ne("")].copy()
    if labelled["role_label"].nunique() < 2:
        raise ValueError("at least two role labels are required for classification")

    group_counts = labelled.groupby("role_label")["company"].nunique()
    splits = min(5, int(group_counts.min()))
    if splits < 2:
        raise ValueError("each role needs advertisements from at least two companies")

    labels = labelled["role_label"]
    groups = labelled["company"]
    combined_predictions = _evaluate_text(combine_text(labelled), labels, groups, splits)
    title_predictions = _evaluate_text(labelled["title"], labels, groups, splits)
    classes = sorted(labels.unique())

    metrics: dict[str, object] = {
        "evaluation": "stratified grouped cross-validation by company",
        "folds": splits,
        "samples": int(len(labelled)),
        "companies": int(groups.nunique()),
        "accuracy": round(float(accuracy_score(labels, combined_predictions)), 4),
        "macro_f1": round(float(f1_score(labels, combined_predictions, average="macro")), 4),
        "title_only_macro_f1": round(
            float(f1_score(labels, title_predictions, average="macro")), 4
        ),
        "classes": classes,
        "confusion_matrix": confusion_matrix(labels, combined_predictions, labels=classes).tolist(),
        "per_class": classification_report(
            labels, combined_predictions, labels=classes, output_dict=True, zero_division=0
        ),
        "warning": "Synthetic-data metrics demonstrate the evaluation code only.",
    }

    model = build_classifier()
    model.fit(combine_text(labelled), labels)
    predictions = labelled[["job_id", "role_label"]].copy()
    predictions["predicted_role"] = combined_predictions
    predictions["correct"] = predictions["role_label"] == predictions["predicted_role"]
    return ModelResult(model=model, metrics=metrics, predictions=predictions)


def predict_role(model: Pipeline, title: str, description: str) -> tuple[str, float]:
    text = pd.Series([f"{title} [title] {description}"])
    label = str(model.predict(text)[0])
    probabilities = model.predict_proba(text)[0]
    confidence = float(np.max(probabilities))
    return label, confidence


def save_model(model: Pipeline, path) -> None:
    joblib.dump(model, path)


def load_model(path) -> Pipeline:
    return joblib.load(path)
