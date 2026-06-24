"""Data-mining methods for job-family discovery and skill bundles."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class ClusterMiningResult:
    assignments: pd.DataFrame
    summary: pd.DataFrame


def _combined_text(jobs: pd.DataFrame) -> pd.Series:
    return (
        jobs["title"].fillna("").astype(str)
        + " [title] "
        + jobs["description"].fillna("").astype(str)
    )


def _top_terms_for_centers(
    centers, feature_names: list[str], *, terms_per_cluster: int
) -> dict[int, str]:
    top_terms: dict[int, str] = {}
    for cluster_id, center in enumerate(centers):
        indexes = center.argsort()[::-1][:terms_per_cluster]
        terms = [feature_names[index] for index in indexes if center[index] > 0]
        top_terms[cluster_id] = ", ".join(terms)
    return top_terms


def _mode_or_empty(series: pd.Series) -> str:
    clean = series.dropna().astype(str)
    clean = clean[clean.ne("")]
    if clean.empty:
        return ""
    return str(clean.value_counts().index[0])


def cluster_job_texts(
    jobs: pd.DataFrame,
    *,
    n_clusters: int = 6,
    max_features: int = 3_000,
    terms_per_cluster: int = 8,
    random_state: int = 42,
) -> ClusterMiningResult:
    """Cluster job excerpts with TF-IDF + KMeans and return private assignments."""

    required = {"job_id", "title", "description"}
    missing = sorted(required - set(jobs.columns))
    if missing:
        raise ValueError(f"jobs are missing columns: {', '.join(missing)}")
    if n_clusters < 1 or max_features < 10 or terms_per_cluster < 1:
        raise ValueError("cluster counts, max_features and terms_per_cluster must be positive")

    frame = jobs.drop_duplicates("job_id").reset_index(drop=True).copy()
    if frame.empty:
        empty_assignments = pd.DataFrame(
            columns=["job_id", "cluster_id", "cluster_distance"]
        )
        empty_summary = pd.DataFrame(
            columns=[
                "cluster_id",
                "job_count",
                "share_of_jobs",
                "top_terms",
                "top_role",
                "top_seniority",
                "top_city",
            ]
        )
        return ClusterMiningResult(empty_assignments, empty_summary)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
        max_features=max_features,
    )
    matrix = vectorizer.fit_transform(_combined_text(frame))
    actual_clusters = min(n_clusters, len(frame))
    if actual_clusters == 1:
        cluster_ids = [0] * len(frame)
        distances = [0.0] * len(frame)
        top_terms = {
            0: ", ".join(vectorizer.get_feature_names_out()[:terms_per_cluster])
        }
    else:
        model = KMeans(n_clusters=actual_clusters, random_state=random_state, n_init="auto")
        cluster_ids = model.fit_predict(matrix).tolist()
        distances = [
            float(row[cluster_id])
            for row, cluster_id in zip(model.transform(matrix), cluster_ids, strict=True)
        ]
        top_terms = _top_terms_for_centers(
            model.cluster_centers_,
            list(vectorizer.get_feature_names_out()),
            terms_per_cluster=terms_per_cluster,
        )

    assignments = pd.DataFrame(
        {
            "job_id": frame["job_id"].astype(str),
            "cluster_id": cluster_ids,
            "cluster_distance": [round(value, 6) for value in distances],
        }
    )
    enriched = frame.merge(assignments, on="job_id", how="inner")
    role_column = "analysis_role" if "analysis_role" in enriched.columns else "role_label"
    if role_column not in enriched.columns:
        enriched[role_column] = ""
    if "seniority" not in enriched.columns:
        enriched["seniority"] = ""
    if "city" not in enriched.columns:
        enriched["city"] = ""

    summary_rows = []
    total = len(enriched)
    for cluster_id, group in enriched.groupby("cluster_id", sort=True):
        summary_rows.append(
            {
                "cluster_id": int(cluster_id),
                "job_count": int(len(group)),
                "share_of_jobs": round(len(group) / total, 4),
                "top_terms": top_terms.get(int(cluster_id), ""),
                "top_role": _mode_or_empty(group[role_column]),
                "top_seniority": _mode_or_empty(group["seniority"]),
                "top_city": _mode_or_empty(group["city"]),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["job_count", "cluster_id"], ascending=[False, True]
    )
    return ClusterMiningResult(assignments, summary.reset_index(drop=True))


def mine_skill_association_rules(
    skill_mentions: pd.DataFrame,
    *,
    total_jobs: int | None = None,
    min_support: float = 0.03,
    min_confidence: float = 0.25,
    max_rules: int = 100,
) -> pd.DataFrame:
    """Mine pairwise skill association rules from job-level skill transactions."""

    required = {"job_id", "skill"}
    missing = sorted(required - set(skill_mentions.columns))
    if missing:
        raise ValueError(f"skill_mentions are missing columns: {', '.join(missing)}")
    if not 0 < min_support <= 1 or not 0 < min_confidence <= 1 or max_rules < 1:
        raise ValueError("support/confidence must be in (0, 1] and max_rules positive")

    transactions = (
        skill_mentions.dropna(subset=["job_id", "skill"])
        .assign(job_id=lambda frame: frame["job_id"].astype(str))
        .assign(skill=lambda frame: frame["skill"].astype(str))
        .groupby("job_id")["skill"]
        .agg(lambda values: frozenset(value for value in values if value))
    )
    transactions = transactions[transactions.map(len).ge(2)]
    denominator = int(total_jobs or skill_mentions["job_id"].nunique())
    if denominator < 1 or transactions.empty:
        return pd.DataFrame(
            columns=[
                "antecedent",
                "consequent",
                "job_count",
                "antecedent_count",
                "consequent_count",
                "support",
                "confidence",
                "lift",
            ]
        )

    single_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for skills in transactions:
        single_counts.update(skills)
        pair_counts.update(combinations(sorted(skills), 2))

    rows = []
    for (first, second), pair_count in pair_counts.items():
        support = pair_count / denominator
        if support < min_support:
            continue
        for antecedent, consequent in ((first, second), (second, first)):
            antecedent_count = single_counts[antecedent]
            consequent_count = single_counts[consequent]
            confidence = pair_count / antecedent_count
            if confidence < min_confidence:
                continue
            consequent_support = consequent_count / denominator
            lift = confidence / consequent_support if consequent_support else 0.0
            rows.append(
                {
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "job_count": int(pair_count),
                    "antecedent_count": int(antecedent_count),
                    "consequent_count": int(consequent_count),
                    "support": round(support, 4),
                    "confidence": round(confidence, 4),
                    "lift": round(lift, 4),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["lift", "confidence", "support"], ascending=[False, False, False])
        .head(max_rules)
        .reset_index(drop=True)
        if rows
        else pd.DataFrame(
            columns=[
                "antecedent",
                "consequent",
                "job_count",
                "antecedent_count",
                "consequent_count",
                "support",
                "confidence",
                "lift",
            ]
        )
    )
