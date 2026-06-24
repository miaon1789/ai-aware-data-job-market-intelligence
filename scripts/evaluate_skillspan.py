#!/usr/bin/env python3
"""Evaluate dictionary coverage on the public SkillSpan test sentences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from job_market_intelligence.benchmarks import (  # noqa: E402
    evaluate_skillspan_sentence_detection,
    load_skillspan,
)
from scripts.fetch_skillspan import COMMIT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/external/skillspan/test.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/skillspan_metrics.json"
    )
    args = parser.parse_args()
    if not args.input.exists():
        parser.error("SkillSpan test data are missing; run scripts/fetch_skillspan.py")

    report = {
        "benchmark": "SkillSpan",
        "commit": COMMIT,
        "license": "MIT",
        "task": "sentence-level detection of any annotated skill or knowledge span",
        "metrics": evaluate_skillspan_sentence_detection(load_skillspan(args.input)),
        "limitations": [
            "SkillSpan is not an Australian labour-market sample.",
            "This is sentence-level detection, not exact span extraction evaluation.",
            "This benchmark does not evaluate required/preferred/negated context.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
