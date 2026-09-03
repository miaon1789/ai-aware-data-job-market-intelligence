#!/usr/bin/env python3
"""Evaluate the query router, separating what it was tuned on from what it was not.

The router makes two decisions with different consequences. Detecting an
aggregation question decides whether the answer comes from SQL or from
retrieval; getting it wrong risks a fabricated number. Distinguishing lexical
from semantic decides which retrieval branch runs; getting it wrong costs some
ranking quality. They are reported separately because they fail independently.

Every accuracy is published with a Wilson interval. Probe sets this small
produce perfect scores easily, and a perfect score on twelve items establishes
much less than it appears to.
"""

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

from job_market_intelligence.rag.routing import classify_query  # noqa: E402
from job_market_intelligence.retrieval.evaluation import wilson_interval  # noqa: E402

REPORTS_ROOT = (ROOT / "reports").resolve()


def load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def score(probes: list[dict], classes: set[str] | None = None) -> dict[str, object]:
    selected = [p for p in probes if classes is None or p["expected_class"] in classes]
    correct = sum(classify_query(p["question"]) == p["expected_class"] for p in selected)
    low, high = wilson_interval(correct, len(selected))
    return {
        "correct": correct,
        "trials": len(selected),
        "accuracy": round(correct / len(selected), 4) if selected else None,
        "wilson_95_low": round(low, 4),
        "wilson_95_high": round(high, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuned", type=Path, default=ROOT / "eval/golden/routing_probes.jsonl")
    parser.add_argument("--heldout", type=Path, default=ROOT / "eval/golden/routing_heldout.jsonl")
    parser.add_argument(
        "--ablation-summary", type=Path, default=ROOT / "reports/retrieval/ablation_summary.json"
    )
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports/retrieval")
    args = parser.parse_args()

    resolved = args.reports_dir.resolve()
    if resolved != REPORTS_ROOT and REPORTS_ROOT not in resolved.parents:
        parser.error("routing reports must stay under reports/")

    tuned, heldout = load(args.tuned), load(args.heldout)
    branch = {"lexical", "semantic"}
    report = {
        "note": (
            "routing_probes.jsonl was used to tune the router's patterns and is "
            "reported for completeness only. routing_heldout.jsonl was written "
            "afterwards and never tuned on; it is the honest estimate."
        ),
        "held_out": {
            "overall": score(heldout),
            "aggregation_detection": score(heldout, {"aggregation"}),
            "branch_choice_lexical_vs_semantic": score(heldout, branch),
        },
        "tuned_on": {
            "overall": score(tuned),
            "branch_choice_lexical_vs_semantic": score(tuned, branch),
        },
        "pooled_tuned_and_held_out": {
            "branch_choice_lexical_vs_semantic": score(tuned + heldout, branch),
        },
    }

    if args.ablation_summary.exists():
        break_even = json.loads(args.ablation_summary.read_text(encoding="utf-8")).get(
            "routing_break_even", {}
        )
        threshold = break_even.get("break_even_router_accuracy")
        if threshold is not None:
            held = report["held_out"]["branch_choice_lexical_vs_semantic"]
            pooled = report["pooled_tuned_and_held_out"][
                "branch_choice_lexical_vs_semantic"
            ]
            report["comparison_with_break_even"] = {
                "break_even_router_accuracy": threshold,
                "held_out_lower_bound": held["wilson_95_low"],
                "held_out_lower_bound_clears_break_even": held["wilson_95_low"] > threshold,
                "pooled_lower_bound": pooled["wilson_95_low"],
                "pooled_lower_bound_clears_break_even": pooled["wilson_95_low"] > threshold,
                "verdict": (
                    "The point estimate clears the break-even but the held-out "
                    "interval does not. Routing is the better bet on this evidence, "
                    "not an established result; a larger lexical/semantic probe set "
                    "is the cheapest way to settle it."
                ),
            }

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "routing_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
