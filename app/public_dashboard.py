"""Public dashboard that can only consume aggregate release tables."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA = Path(os.environ.get("PUBLIC_DATA_DIR", ROOT / "data/public"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", ROOT / "reports"))

st.set_page_config(page_title="Australian Data Job Market Aggregates", layout="wide")
st.title("Australian Data Job Market Aggregates")

required_files = {
    "market": PUBLIC_DATA / "market_summary.csv",
    "seniority": PUBLIC_DATA / "seniority_summary.csv",
    "skills": PUBLIC_DATA / "skill_summary.csv",
    "metadata": PUBLIC_DATA / "metadata.json",
}
missing = [str(path.name) for path in required_files.values() if not path.exists()]
if missing:
    st.error(
        "Aggregate release files are missing: "
        + ", ".join(missing)
        + ". Run `python scripts/build_public_release.py`."
    )
    st.stop()

metadata = json.loads(required_files["metadata"].read_text(encoding="utf-8"))
market = pd.read_csv(required_files["market"])
seniority = pd.read_csv(required_files["seniority"])
skills = pd.read_csv(required_files["skills"])


def _optional_csv(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _optional_json(name: str) -> dict[str, object]:
    path = REPORTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


semantic_labels = _optional_csv("semantic_label_distribution.csv")
semantic_role_ai = _optional_csv("semantic_role_ai_signal.csv")
semantic_role_coding = _optional_csv("semantic_role_coding_signal.csv")
semantic_summary = _optional_json("semantic_analysis_summary.json")
audit_summary = _optional_json("human_llm_audit_annotation_quality.json")
adjudication_summary = _optional_json("adjudicated_annotation_summary.json")
data_quality = _optional_json("data_quality.json")

if metadata["data_kind"] == "synthetic":
    st.warning("Synthetic demonstration data: these charts are not labour-market evidence.")
else:
    st.info(metadata["attribution"])
st.caption(metadata["disclaimer"])
if metadata.get("description_truncation_detected"):
    st.warning(
        "The API description field is truncated. Skill charts measure mentions in API "
        "excerpts and cannot represent complete job requirements."
    )
if (
    metadata.get("role_label_source")
    not in {"human labels", "human-adjudicated semantic labels"}
    and not adjudication_summary
):
    st.warning(
        "Role groups are provisional and come from an unvalidated demonstration baseline. "
        "Do not interpret role comparisons as validated market estimates."
    )
elif adjudication_summary:
    st.success(
        "Semantic label reports use human-reviewed labels, Claude-assisted audit and "
        "manual adjudication of model disagreements."
    )

if market.empty:
    st.warning(
        "No market cells met the publication threshold. Collect more advertisements or "
        "use broader, pre-specified groups; do not lower the real-data threshold below 10."
    )
    st.stop()

with st.sidebar:
    st.header("Aggregate filters")
    cities = st.multiselect(
        "City", sorted(market["city"].unique()), default=sorted(market["city"].unique())
    )
    roles = st.multiselect(
        "Role", sorted(market["role"].unique()), default=sorted(market["role"].unique())
    )

market_filtered = market[market["city"].isin(cities) & market["role"].isin(roles)]
seniority_filtered = seniority[
    seniority["city"].isin(cities) & seniority["role"].isin(roles)
]
skills_filtered = skills[skills["city"].isin(cities) & skills["role"].isin(roles)]

overview_tab, semantic_tab, skills_tab, audit_tab, governance_tab = st.tabs(
    ["Market overview", "Semantic labels", "Skill explorer", "Audit quality", "Governance"]
)
with overview_tab:
    left, right = st.columns(2)
    left.metric("Published aggregate cells", len(market_filtered))
    right.metric("Minimum cell size", metadata["minimum_cell_size"])
    st.plotly_chart(
        px.bar(
            market_filtered,
            x="role",
            y="advertisement_count",
            color="city",
            facet_col="collection_month",
            barmode="group",
            title="Published advertisements by role, city and month",
        ),
        width="stretch",
    )
    if not seniority_filtered.empty:
        st.plotly_chart(
            px.bar(
                seniority_filtered,
                x="role",
                y="advertisement_count",
                color="seniority",
                facet_col="city",
                barmode="stack",
                title="Published seniority signals",
            ),
            width="stretch",
        )

with semantic_tab:
    if semantic_labels.empty:
        st.info(
            "Semantic aggregate reports are unavailable. Run "
            "`scripts/build_semantic_analysis.py`."
        )
    else:
        st.subheader("Adjudicated semantic labels")
        labelled_jobs = semantic_summary.get("labelled_jobs")
        if labelled_jobs:
            st.caption(
                f"Labelled train/dev sample: {labelled_jobs} advertisements. "
                "All tables are aggregate-only and cell-suppressed where published."
            )
        dimension = st.selectbox(
            "Label dimension",
            sorted(semantic_labels["dimension"].unique()),
            index=sorted(semantic_labels["dimension"].unique()).index("ai_signal")
            if "ai_signal" in set(semantic_labels["dimension"])
            else 0,
        )
        dimension_frame = semantic_labels[semantic_labels["dimension"] == dimension]
        st.plotly_chart(
            px.bar(
                dimension_frame,
                x="job_count",
                y="label",
                orientation="h",
                title=f"{dimension.replace('_', ' ').title()} distribution",
            ).update_layout(yaxis={"categoryorder": "total ascending"}),
            width="stretch",
        )

        left, right = st.columns(2)
        if not semantic_role_ai.empty:
            left.plotly_chart(
                px.bar(
                    semantic_role_ai,
                    x="role_label",
                    y="share_within_row",
                    color="ai_signal_label",
                    title="AI signal by role family",
                    labels={"share_within_row": "Share within role"},
                ),
                width="stretch",
            )
        if not semantic_role_coding.empty:
            right.plotly_chart(
                px.bar(
                    semantic_role_coding,
                    x="role_label",
                    y="share_within_row",
                    color="coding_signal_label",
                    title="Observed coding signal by role family",
                    labels={"share_within_row": "Share within role"},
                ),
                width="stretch",
            )

with skills_tab:
    if skills_filtered.empty:
        st.info("No skill cells met the publication threshold for these filters.")
    else:
        top_skills = (
            skills_filtered.groupby(["skill", "category"], as_index=False)[
                "ads_with_skill"
            ]
            .sum()
            .sort_values("ads_with_skill", ascending=False)
            .head(20)
        )
        st.plotly_chart(
            px.bar(
                top_skills,
                x="ads_with_skill",
                y="skill",
                color="category",
                orientation="h",
                title="Top skills across published aggregate cells",
            ).update_layout(yaxis={"categoryorder": "total ascending"}),
            width="stretch",
        )

with audit_tab:
    if not audit_summary or not adjudication_summary:
        st.info("Audit reports are unavailable. Run the LLM audit and adjudication scripts.")
    else:
        document_audit = audit_summary.get("document", {})
        agreement = document_audit.get("agreement", {})
        left, middle, right = st.columns(3)
        left.metric("LLM-audited pairs", agreement.get("paired_annotations", "n/a"))
        middle.metric("Exact pair agreement", agreement.get("exact_pair_agreement", "n/a"))
        right.metric("Adjudicated rows", adjudication_summary.get("adjudication_rows", "n/a"))

        st.subheader("Cohen's kappa by dimension")
        kappa_rows = [
            {"dimension": "role", "kappa": agreement.get("role_cohen_kappa")},
            {"dimension": "entry_fit", "kappa": agreement.get("entry_fit_cohen_kappa")},
            {"dimension": "ai_signal", "kappa": agreement.get("ai_signal_cohen_kappa")},
            {
                "dimension": "coding_signal",
                "kappa": agreement.get("coding_signal_cohen_kappa"),
            },
        ]
        kappa = pd.DataFrame(kappa_rows).dropna()
        if not kappa.empty:
            st.plotly_chart(
                px.bar(kappa, x="dimension", y="kappa", range_y=[0, 1]),
                width="stretch",
            )

        st.subheader("Manual adjudication changes")
        changes = adjudication_summary.get("changes_by_dimension", {})
        if isinstance(changes, dict) and changes:
            changes_frame = pd.DataFrame(
                [
                    {"dimension": key.replace("_label", ""), "changed_tasks": value}
                    for key, value in changes.items()
                ]
            )
            st.plotly_chart(
                px.bar(changes_frame, x="dimension", y="changed_tasks"),
                width="stretch",
            )

with governance_tab:
    st.write(metadata["attribution"])
    st.write(f"Data kind: `{metadata['data_kind']}`")
    st.write(f"Collection period: `{metadata['collection_month_start']}` to "
             f"`{metadata['collection_month_end']}`")
    st.write(f"Minimum published cell size: `{metadata['minimum_cell_size']}`")
    st.write(f"Text scope: `{metadata.get('text_scope', 'not recorded')}`")
    st.write(f"Role label source: `{metadata.get('role_label_source', 'not recorded')}`")
    if data_quality:
        st.write(f"Clean private rows: `{data_quality.get('clean_rows', 'not recorded')}`")
        st.write(
            "Description exactly 500 characters share: "
            f"`{data_quality.get('description_exactly_500_share', 'not recorded')}`"
        )
    if adjudication_summary:
        st.write(
            "Semantic label source: `human-reviewed labels + Claude audit + manual "
            "adjudication`"
        )
    st.write("Contains row-level records: `false`")
    st.write("Contains job descriptions: `false`")
    if metadata.get("source_url"):
        st.markdown(f"Source: [{metadata['source']}]({metadata['source_url']})")
    if metadata.get("terms_url"):
        st.markdown(f"Terms checked: [Adzuna API Terms]({metadata['terms_url']})")
