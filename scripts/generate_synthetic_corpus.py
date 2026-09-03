#!/usr/bin/env python3
"""Generate a fictional full-text corpus so the retrieval pipeline is reproducible.

The real corpus cannot be published: it is collected under API terms that keep
the advertisement text private. Without a substitute, none of the retrieval
code in this repository could be run by anyone else. This generator produces
structurally realistic advertisements -- headed sections, requirement bullets,
benefits boilerplate -- naming the same tools the lexical golden queries use,
so `build_chunks.py` through `run_retrieval_ablation.py` all execute end to end
on data that is safe to commit.

Every company, person and detail here is invented. The numbers it produces are
a smoke test of the pipeline, not a finding about the labour market.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

COMPANIES = [
    ("Harbour Metrics", "retail analytics"),
    ("Wattle Models", "insurance pricing"),
    ("Koala Freight", "logistics"),
    ("Banksia Health", "clinical research"),
    ("Laneway Energy", "renewable energy"),
    ("Quay Financial", "payments"),
    ("Southern Grid", "utilities"),
    ("Bluegum Media", "streaming"),
]
CITIES = ["Sydney", "Melbourne"]

ROLE_TEMPLATES = {
    "Data Engineer": {
        "titles": ["Data Engineer", "Senior Data Engineer", "Graduate Data Engineer"],
        "responsibilities": [
            "Build and maintain batch pipelines that land source systems into the warehouse",
            "Model curated tables with dbt and review incoming pull requests",
            "Schedule and monitor workloads in Airflow, and page-tune the alerts",
            "Partner with analysts to agree contracts for the tables they depend on",
            "Move ingestion from Fivetran onto in-house connectors where cost demands it",
        ],
        "requirements": [
            "Three or more years writing production Python and SQL",
            "Hands-on experience with Snowflake, BigQuery or Redshift",
            "Comfort with Terraform and deploying through CI",
            "Working knowledge of Spark or another distributed processing engine",
            "Experience operating Kafka or an equivalent streaming platform",
        ],
        "preferred": [
            "Exposure to Databricks and Unity Catalog",
            "Familiarity with Kubernetes and container-based deployment",
            "Prior work with Grafana or Datadog for pipeline observability",
        ],
    },
    "Data Analyst": {
        "titles": ["Data Analyst", "Business Intelligence Analyst", "Graduate Data Analyst"],
        "responsibilities": [
            "Answer questions from the commercial team and explain what the numbers mean",
            "Build and maintain Power BI dashboards used by the leadership group",
            "Translate vague business questions into something a query can answer",
            "Present findings to people with no technical background and agree next steps",
            "Keep definitions consistent so two teams do not report different revenue",
        ],
        "requirements": [
            "Strong SQL and the judgement to know when a number looks wrong",
            "Experience with Power BI, Tableau, Looker or Metabase",
            "Clear written communication for a non-technical audience",
            "Comfort working with incomplete data and saying so out loud",
        ],
        "preferred": [
            "Python for analysis, or a willingness to learn it here",
            "Experience running A/B tests and interpreting the results",
            "Exposure to dbt for managing shared metric definitions",
        ],
    },
    "Data Scientist": {
        "titles": ["Data Scientist", "Machine Learning Engineer", "Applied Scientist"],
        "responsibilities": [
            "Frame business problems as measurable modelling questions",
            "Design and run controlled experiments to establish whether a change helped",
            "Train, evaluate and monitor models that run in production",
            "Track experiments with MLflow so results can be reproduced months later",
            "Explain model behaviour to stakeholders who will act on its output",
        ],
        "requirements": [
            "Solid grounding in statistics and experimental design",
            "Python with scikit-learn, PyTorch or TensorFlow",
            "Experience taking a model past the notebook and into a service",
            "SQL fluent enough to build your own features",
        ],
        "preferred": [
            "Experience with large language models and retrieval-augmented generation",
            "Familiarity with Sagemaker, Vertex AI or Databricks ML",
            "Published or open-source work we can read",
        ],
    },
    "AI / Automation": {
        "titles": ["AI Engineer", "Automation Engineer", "AI Solutions Engineer"],
        "responsibilities": [
            "Build assistants that take actions on behalf of internal teams",
            "Design retrieval pipelines over internal documents and measure their quality",
            "Replace repetitive manual processes with reliable automated workflows",
            "Write evaluation sets so a prompt change can be judged rather than guessed at",
            "Work with security and compliance on what data a model may see",
        ],
        "requirements": [
            "Strong Python and comfort designing and consuming APIs",
            "Experience with LangChain, LlamaIndex or an equivalent framework",
            "Prompt engineering grounded in evaluation rather than intuition",
            "Understanding of vector search, chunking and reranking trade-offs",
        ],
        "preferred": [
            "Experience running background jobs and queues in production",
            "Exposure to Zapier, n8n or another workflow automation tool",
            "Familiarity with TypeScript for the surrounding application layer",
        ],
    },
}

BENEFITS = [
    "Hybrid working, with two days a week in the office",
    "Additional paid leave over the December shutdown",
    "A yearly budget for conferences and training",
    "Employee share scheme after twelve months",
    "Paid parental leave for all parents regardless of gender",
]
LEGAL = (
    "{company} is an equal opportunity employer. We welcome applications from "
    "candidates of every background, and we will make reasonable adjustments at "
    "any stage of the process on request. We never ask candidates for payment, "
    "and we will only contact you from a {slug}.example address."
)
INTRO = (
    "{company} is a fictional {sector} company created for testing this "
    "repository's retrieval pipeline. We are a team of roughly {size} people "
    "based in {city}, and none of us exist."
)


def build_description(rng: random.Random, role: str, company: str, sector: str, city: str) -> str:
    template = ROLE_TEMPLATES[role]
    size = rng.choice([40, 80, 120, 250, 400])
    sections = [
        f"About {company}",
        INTRO.format(company=company, sector=sector, size=size, city=city),
        "",
        "About the role",
        (
            f"You will join our data team in {city}, reporting to the head of data. "
            "This advertisement is synthetic and describes no real vacancy."
        ),
        "",
        "What you'll do",
    ]
    sections += [f"- {item}" for item in rng.sample(template["responsibilities"], 4)]
    sections += ["", "Requirements"]
    sections += [f"- {item}" for item in rng.sample(template["requirements"], 3)]
    sections += ["", "Nice to have"]
    sections += [f"- {item}" for item in rng.sample(template["preferred"], 2)]
    sections += ["", "What we offer"]
    sections += [f"- {item}" for item in rng.sample(BENEFITS, 3)]
    sections += [
        "",
        "Equal opportunity",
        LEGAL.format(company=company, slug=company.lower().replace(" ", "")),
    ]
    return "\n".join(sections)


def generate(rows: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    reference = date(2026, 6, 30)
    roles = list(ROLE_TEMPLATES)
    records = []
    for index in range(rows):
        role = roles[index % len(roles)]
        company, sector = COMPANIES[index % len(COMPANIES)]
        city = CITIES[index % len(CITIES)]
        title = rng.choice(ROLE_TEMPLATES[role]["titles"])
        structured = build_description(rng, role, company, sector, city)
        records.append(
            {
                "job_id": f"synthetic-{index:03d}",
                "title": title,
                # The flat copy is what deduplication and skill extraction read;
                # the structured copy is what the chunker splits on.
                "description": " ".join(structured.split()),
                "description_structured": structured,
                "city": city,
                "company": company,
                "posted_at": (reference - timedelta(days=rng.randint(0, 80))).isoformat(),
                "source": "synthetic-corpus",
                "source_url": "",
                "role_label": role if index % 3 == 0 else "",
                "collection_query": "synthetic",
                "collection_group": "synthetic",
                "retrieved_at": reference.isoformat(),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/synthetic/retrieval_corpus.csv"
    )
    parser.add_argument("--rows", type=int, default=28)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    frame = generate(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    lengths = frame["description_structured"].str.len()
    print(
        f"wrote {len(frame)} synthetic advertisements to "
        f"{args.output.relative_to(ROOT)}\n"
        f"  mean length {lengths.mean():.0f} characters, "
        f"{frame['description_structured'].str.count(chr(10)).mean():.0f} lines"
    )


if __name__ == "__main__":
    main()
