#!/usr/bin/env python3
"""Generate fictional job ads for reproducible pipeline demonstrations."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ROLE_CONFIG = {
    "Data Analyst": {
        "companies": [
            "Harbour Metrics",
            "Koala Reports",
            "Southern Insights",
            "Civic Lens",
            "Bluegum Retail",
            "Laneway Health",
        ],
        "titles": ["Graduate Data Analyst", "Junior BI Analyst", "Data Analyst"],
        "required": [
            "SQL, Excel and Power BI",
            "SQL, data visualisation and stakeholder communication",
            "Excel, Tableau and data quality",
        ],
        "preferred": ["Python", "Power BI and Python", "Looker or Tableau"],
        "work": [
            "build recurring dashboards and explain performance trends",
            "validate operational reports and translate questions into analysis",
            "write queries and communicate actionable insights to stakeholders",
        ],
    },
    "Data Scientist": {
        "companies": [
            "Wattle Models",
            "Signal Quay",
            "Banksia AI",
            "Quant Cove",
            "Orbit Research",
            "Yarra Forecasting",
        ],
        "titles": ["Graduate Data Scientist", "Junior Machine Learning Analyst", "Data Scientist"],
        "required": [
            "Python, SQL, statistics and machine learning",
            "Python, scikit-learn and statistical modelling",
            "SQL, Python and machine learning",
        ],
        "preferred": ["AWS and Docker", "PyTorch or TensorFlow", "NLP and generative AI"],
        "work": [
            "design experiments and evaluate predictive models",
            "prepare features and explain model performance to stakeholders",
            "prototype machine learning solutions and monitor data quality",
        ],
    },
    "Data Engineer": {
        "companies": [
            "Pipeline Point",
            "Cloud Gum",
            "Federation Data",
            "Quarry Systems",
            "River ETL",
            "Tramline Platforms",
        ],
        "titles": ["Graduate Data Engineer", "Junior Analytics Engineer", "Data Engineer"],
        "required": [
            "Python, SQL, ETL and data modelling",
            "SQL, dbt and data pipelines",
            "Python, Spark and data quality",
        ],
        "preferred": ["AWS and Airflow", "Snowflake or Databricks", "Docker and Kubernetes"],
        "work": [
            "build tested batch pipelines and reliable warehouse models",
            "integrate APIs and monitor data platform quality",
            "develop scalable transformations and maintain technical documentation",
        ],
    },
}


def generate_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    start = date(2026, 4, 1)
    sequence = 1
    for role_index, (role, config) in enumerate(ROLE_CONFIG.items()):
        for company_index, company in enumerate(config["companies"]):
            city = "Sydney" if company_index % 2 == 0 else "Melbourne"
            for variant in range(3):
                title = config["titles"][variant]
                seniority = (
                    "This is an entry-level opportunity for a graduate with "
                    "0-2 years of experience."
                    if variant < 2
                    else "Applicants should bring 3 years of relevant commercial experience."
                )
                description = (
                    f"{company} is hiring in {city}. You will {config['work'][variant]}. "
                    f"Essential skills are {config['required'][variant]}. {seniority} "
                    f"Experience with {config['preferred'][variant]} is desirable. "
                    "You will use Git and problem solving in an agile team."
                )
                records.append(
                    {
                        "job_id": f"SYN-{sequence:04d}",
                        "title": title,
                        "description": description,
                        "city": city,
                        "company": company,
                        "posted_at": str(start + timedelta(days=sequence + role_index * 2)),
                        "source": "synthetic-demo",
                        "source_url": "",
                        "role_label": role,
                    }
                )
                sequence += 1

    # Deliberate cross-source copies exercise the duplicate audit.
    for original in records[:3]:
        copied = original.copy()
        copied["job_id"] = f"SYN-{sequence:04d}"
        copied["posted_at"] = str(date.fromisoformat(original["posted_at"]) + timedelta(days=1))
        copied["source"] = "synthetic-demo-copy"
        records.append(copied)
        sequence += 1
    return records


def generate_csv(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(generate_records()).to_csv(output, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/synthetic/job_ads.csv")
    args = parser.parse_args()
    path = generate_csv(args.output)
    print(f"Wrote {len(generate_records())} synthetic records to {path}")


if __name__ == "__main__":
    main()
