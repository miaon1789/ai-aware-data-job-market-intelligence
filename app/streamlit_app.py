"""Interactive market explorer and job-description comparison."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_market_intelligence.modelling import load_model, predict_role  # noqa: E402
from job_market_intelligence.seniority import classify_seniority  # noqa: E402
from job_market_intelligence.skills import SkillExtractor  # noqa: E402

DATABASE = Path(
    os.environ.get("PRIVATE_DATABASE", ROOT / "data/processed/job_market.duckdb")
)
MODEL_PATH = Path(
    os.environ.get("PRIVATE_MODEL_PATH", ROOT / "data/processed/role_classifier.joblib")
)
METRICS_PATH = Path(
    os.environ.get("PRIVATE_METRICS_PATH", ROOT / "reports/model_metrics.json")
)

st.set_page_config(page_title="Australian Data Job Market Intelligence", layout="wide")


@st.cache_data
def read_table(table: str) -> pd.DataFrame:
    allowed = {"jobs", "skill_mentions", "duplicate_audit", "role_city_summary"}
    if table not in allowed:
        raise ValueError("unsupported table")
    with duckdb.connect(str(DATABASE), read_only=True) as connection:
        return connection.execute(f"SELECT * FROM {table}").fetchdf()


@st.cache_resource
def resources():
    return load_model(MODEL_PATH), SkillExtractor()


st.title("Australian Entry-Level Data Job Market Intelligence")

if not DATABASE.exists() or not MODEL_PATH.exists():
    st.error("Pipeline outputs are missing. Run `python scripts/run_pipeline.py` first.")
    st.stop()

jobs = read_table("jobs")
mentions = read_table("skill_mentions")
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
if jobs["source"].astype(str).str.startswith("synthetic").all():
    st.warning(
        "Demo mode: the bundled records are synthetic. Charts and model scores are not "
        "evidence about the Australian labour market."
    )
else:
    st.warning(
        "Private analyst mode: this app loads row-level processed data and must not be "
        "deployed publicly. Use `app/public_dashboard.py` for aggregate publication."
    )
    st.caption("Vacancy data sourced from The Adzuna API: https://www.adzuna.com.au/")

with st.sidebar:
    st.header("Sample filters")
    selected_cities = st.multiselect(
        "City", sorted(jobs["city"].unique()), default=sorted(jobs["city"].unique())
    )
    selected_roles = st.multiselect(
        "Role",
        sorted(jobs["analysis_role"].unique()),
        default=sorted(jobs["analysis_role"].unique()),
    )
    selected_seniority = st.multiselect(
        "Seniority", sorted(jobs["seniority"].unique()), default=sorted(jobs["seniority"].unique())
    )

filtered = jobs[
    jobs["city"].isin(selected_cities)
    & jobs["analysis_role"].isin(selected_roles)
    & jobs["seniority"].isin(selected_seniority)
]
filtered_mentions = mentions[mentions["job_id"].isin(filtered["job_id"])]

overview_tab, skills_tab, comparison_tab, methods_tab = st.tabs(
    ["Market overview", "Skill explorer", "JD comparison", "Methods"]
)

with overview_tab:
    first, second, third, fourth = st.columns(4)
    first.metric("Advertisements", len(filtered))
    second.metric("Companies", filtered["company"].nunique())
    third.metric("Normalised skills", filtered_mentions["skill"].nunique())
    fourth.metric("Duplicates removed", len(read_table("duplicate_audit")))

    counts = (
        filtered.groupby(["city", "analysis_role"], as_index=False)
        .size()
        .rename(columns={"size": "advertisements", "analysis_role": "role"})
    )
    if counts.empty:
        st.info("No advertisements match the selected filters.")
    else:
        st.plotly_chart(
            px.bar(
                counts,
                x="role",
                y="advertisements",
                color="city",
                barmode="group",
                title="Advertisements by role and city",
            ),
            width="stretch",
        )

with skills_tab:
    denominator = filtered["job_id"].nunique()
    if denominator == 0:
        st.info("No advertisements match the selected filters.")
    else:
        skill_counts = (
            filtered_mentions.groupby(["skill", "category"], as_index=False)["job_id"]
            .nunique()
            .rename(columns={"job_id": "ads_with_skill"})
        )
        skill_counts["share"] = skill_counts["ads_with_skill"] / denominator
        skill_counts = skill_counts.sort_values(
            ["ads_with_skill", "skill"], ascending=[False, True]
        ).head(20)
        st.plotly_chart(
            px.bar(
                skill_counts,
                x="share",
                y="skill",
                color="category",
                orientation="h",
                hover_data=["ads_with_skill"],
                title=f"Top skills in filtered sample (n={denominator} advertisements)",
            ).update_layout(yaxis={"categoryorder": "total ascending"}),
            width="stretch",
        )
        context_counts = (
            filtered_mentions.groupby(["context"], as_index=False)
            .size()
            .rename(columns={"size": "mentions"})
        )
        st.dataframe(context_counts, hide_index=True, width="stretch")

with comparison_tab:
    st.subheader("Compare one advertisement with this sample")
    st.caption(
        "Text is processed in memory for this session and is not written to the database. "
        "The comparison evaluates advertisement language, not a candidate."
    )
    title = st.text_input("Job title", placeholder="Graduate Data Analyst")
    city = st.selectbox("Comparison city", ["Sydney", "Melbourne"])
    description = st.text_area("Paste a job description", height=220)
    if st.button("Analyse description", type="primary"):
        if len(description.strip()) < 20:
            st.error("Enter at least 20 characters of job-description text.")
        else:
            model, extractor = resources()
            role, model_score = predict_role(model, title, description)
            seniority, evidence = classify_seniority(title, description)
            extracted = extractor.extract(description)
            extracted_names = {
                mention.skill for mention in extracted if mention.context != "negated"
            }

            benchmark_jobs = jobs[
                (jobs["analysis_role"] == role) & (jobs["city"] == city)
            ]
            benchmark_mentions = mentions[mentions["job_id"].isin(benchmark_jobs["job_id"])]
            benchmark = (
                benchmark_mentions.groupby("skill")["job_id"]
                .nunique()
                .sort_values(ascending=False)
            )
            benchmark_share = benchmark / max(1, benchmark_jobs["job_id"].nunique())

            left, middle, right = st.columns(3)
            left.metric("Predicted role", role)
            middle.metric("Model score", f"{model_score:.0%}")
            right.metric("Seniority signal", seniority)
            st.caption(
                "The model score is the baseline classifier's largest probability and is not "
                "a calibrated confidence estimate."
            )
            if evidence:
                st.write(f"Seniority evidence: `{evidence}`")

            extracted_frame = pd.DataFrame(
                [
                    {"skill": item.skill, "category": item.category, "context": item.context}
                    for item in extracted
                ]
            )
            st.markdown("**Extracted advertisement skills**")
            if extracted_frame.empty:
                st.info("No dictionary skills were detected.")
            else:
                st.dataframe(extracted_frame, hide_index=True, width="stretch")

            comparison_rows = [
                {
                    "skill": skill,
                    "share_in_benchmark": f"{share:.0%}",
                    "ads_with_skill": int(benchmark[skill]),
                    "present_in_pasted_ad": skill in extracted_names,
                }
                for skill, share in benchmark_share.head(12).items()
            ]
            st.markdown(
                f"**Descriptive comparison: {role}, {city} "
                f"(n={benchmark_jobs['job_id'].nunique()})**"
            )
            st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")

with methods_tab:
    st.subheader("Evaluation")
    first, second, third = st.columns(3)
    first.metric("Grouped CV macro-F1", f"{metrics['macro_f1']:.3f}")
    second.metric("Title-only macro-F1", f"{metrics['title_only_macro_f1']:.3f}")
    third.metric("Evaluation samples", metrics["samples"])
    st.write(metrics["evaluation"])
    st.warning(metrics["warning"])
    st.markdown(
        "See `docs/DATA_CARD.md` and `docs/ETHICS.md` for provenance requirements, "
        "limitations and prohibited uses."
    )
