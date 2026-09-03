"""Filter merged job ads down to data / IT / ML roles.

Broad job-board queries (e.g. Adzuna "graduate ...") pull in unrelated graduate
roles — nursing, radiography, town planning, marketing. This keeps only roles
that either have a clearly technical title or mention several technical skills,
and drops clearly non-technical professions.
"""

from __future__ import annotations

import re

import pandas as pd

# Every extracted skill category except soft "Professional" skills counts as a
# technical signal (see skills config: Programming, Data Science, Cloud, ...).
TECH_SKILL_CATEGORIES = frozenset(
    {
        "AI",
        "AI Engineering",
        "AI Tooling",
        "Analytics",
        "Automation",
        "BI",
        "Cloud",
        "Data",
        "Data Engineering",
        "Data Science",
        "Engineering",
        "Programming",
    }
)

# Titles that unambiguously denote a data/IT/ML/software role.
_STRONG_TECH = re.compile(
    r"\b("
    r"software|developer|programmer|devops|mlops|sre|"
    r"data analyst|data scientist|data science|data engineer|data engineering|"
    r"data analytics|analytics engineer|analytics|machine learning|deep learning|"
    r"artificial intelligence|ml engineer|ai engineer|ai/ml|ml/ai|"
    r"\bai\b|\bml\b|gen ?ai|\bllm\b|databricks|snowflake|"
    r"business intelligence|bi analyst|bi developer|"
    r"computer vision|natural language|nlp|"
    r"\bcloud\b|cyber ?security|information technology|it support|it graduate|"
    r"back[ -]?end|front[ -]?end|full[ -]?stack|"
    r"site reliability|platform engineer|cloud engineer|data platform|"
    r"database administrator|dba|"
    r"software architect|solutions? architect|data architect|cloud architect|"
    r"enterprise architect|technical architect|"
    r"qa engineer|test engineer|automation engineer|security engineer|"
    r"systems engineer|network engineer|web developer|application developer|"
    r"data warehouse|etl"
    r")\b"
)

# Non-technical professions to drop when the title is not clearly technical.
_NON_TECH = re.compile(
    r"\b("
    r"nurse|nursing|midwife|radiograph|radiologist|sonograph|patholog|"
    r"therapist|therapy|occupational|physiotherap|physio|dental|dentist|"
    r"pharmac|psycholog|psychiatr|counsell|clinician|clinical|physician|"
    r"paramedic|dietit|nutrition|optometr|chiropract|podiatr|audiolog|speech|"
    r"veterinar|teacher|educator|tutor|lecturer|childcare|"
    r"chef|cook|barista|hospitality|waiter|"
    r"electrician|plumber|plumbing|carpenter|mechanic|welder|boilermaker|"
    r"lawyer|solicitor|barrister|paralegal|"
    r"accountant|bookkeep|payroll|"
    r"town planner|urban planner|environmental planner|surveyor|"
    r"marketing|sales representative|sales development|sales consultant|"
    r"account executive|real estate|property manager|"
    r"social work|community worker|support worker|disability|aged care|"
    r"gardener|landscap|cleaner|forklift|"
    r"pathology|phlebotom|occupational therap"
    r")\b"
)


def is_relevant_title(title: str, tech_skill_count: int, *, min_tech_skills: int = 2) -> bool:
    """Decide whether a role is data/IT/ML relevant.

    A clearly technical title is always kept; a clearly non-technical title is
    dropped; otherwise the role is kept only if it mentions enough technical
    skills (``min_tech_skills``).
    """

    text = str(title or "").lower()
    if _STRONG_TECH.search(text):
        return True
    if _NON_TECH.search(text):
        return False
    return tech_skill_count >= min_tech_skills


def filter_relevant_jobs(
    jobs: pd.DataFrame,
    skill_mentions: pd.DataFrame,
    *,
    min_tech_skills: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (kept_jobs, filtered_skill_mentions, dropped_audit).

    ``dropped_audit`` lists the removed roles with their title for review.
    """

    if "job_id" not in jobs.columns or "title" not in jobs.columns:
        raise ValueError("jobs must contain job_id and title columns")

    tech_counts: dict[str, int] = {}
    if not skill_mentions.empty and {"job_id", "category"} <= set(skill_mentions.columns):
        tech = skill_mentions[skill_mentions["category"].isin(TECH_SKILL_CATEGORIES)]
        tech_counts = tech.groupby(tech["job_id"].astype(str)).size().to_dict()

    keep_mask = jobs.apply(
        lambda row: is_relevant_title(
            row["title"],
            tech_counts.get(str(row["job_id"]), 0),
            min_tech_skills=min_tech_skills,
        ),
        axis=1,
    )
    kept = jobs[keep_mask].reset_index(drop=True)
    dropped = (
        jobs.loc[~keep_mask, ["job_id", "title"]]
        .assign(reason="not data/IT/ML relevant")
        .reset_index(drop=True)
    )
    kept_ids = set(kept["job_id"].astype(str))
    if skill_mentions.empty:
        filtered_mentions = skill_mentions
    else:
        filtered_mentions = skill_mentions[
            skill_mentions["job_id"].astype(str).isin(kept_ids)
        ].reset_index(drop=True)
    return kept, filtered_mentions, dropped
