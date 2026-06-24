"""LLM-assisted document-label auditing helpers.

The LLM is used as a review aid, not as ground-truth annotation.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

ROLE_VALUES = {
    "Data Analyst / BI",
    "Data Engineer",
    "Data Scientist / ML",
    "AI / Automation",
    "Other / Mixed",
}
ENTRY_FIT_VALUES = {"Likely entry-level", "Experienced role", "Unclear"}
AI_SIGNAL_VALUES = {
    "Explicit GenAI / LLM tool",
    "Explicit AI / ML system work",
    "Workflow automation",
    "AI-adjacent but vague",
    "No AI signal",
    "Unclear",
}
CODING_SIGNAL_VALUES = {
    "Standalone coding skill",
    "Automation / scripting",
    "Production engineering",
    "No coding signal",
}
CODING_LIKELY_VALUES = {
    "Likely coding-intensive",
    "Likely some analytical scripting",
    "Likely low/no coding",
    "Unclear",
}


@dataclass(frozen=True)
class ParsedLLMAudit:
    role: str
    entry_fit: str
    ai_signal: str
    coding_signal: str
    coding_likely_by_role: str
    confidence: str
    rationale: str

    @property
    def observed_labels(self) -> list[str]:
        return [
            f"ROLE::{self.role}",
            f"ENTRY_FIT::{self.entry_fit}",
            f"AI_SIGNAL::{self.ai_signal}",
            f"CODING_SIGNAL::{self.coding_signal}",
        ]


def _options(values: set[str]) -> str:
    return "\n".join(f"- {value}" for value in sorted(values))


def llm_audit_json_schema() -> dict[str, Any]:
    """Return the structured-output schema for LLM audit responses."""

    return {
        "name": "job_label_audit",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["observed", "inferred", "confidence", "rationale"],
            "properties": {
                "observed": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "role",
                        "entry_fit",
                        "ai_signal",
                        "coding_signal",
                    ],
                    "properties": {
                        "role": {"type": "string", "enum": sorted(ROLE_VALUES)},
                        "entry_fit": {
                            "type": "string",
                            "enum": sorted(ENTRY_FIT_VALUES),
                        },
                        "ai_signal": {
                            "type": "string",
                            "enum": sorted(AI_SIGNAL_VALUES),
                        },
                        "coding_signal": {
                            "type": "string",
                            "enum": sorted(CODING_SIGNAL_VALUES),
                        },
                    },
                },
                "inferred": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["coding_likely_by_role"],
                    "properties": {
                        "coding_likely_by_role": {
                            "type": "string",
                            "enum": sorted(CODING_LIKELY_VALUES),
                        }
                    },
                },
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "rationale": {"type": "string", "minLength": 1},
            },
        },
    }


def build_llm_audit_messages(
    text: str, *, current_labels: list[str] | None = None
) -> list[dict[str, str]]:
    """Build a strict prompt for excerpt-level labels plus role-based inference."""

    current = "\n".join(f"- {label}" for label in sorted(current_labels or []))
    if not current:
        current = "- not provided"
    system = (
        "You are auditing labels for private job-ad excerpt research. "
        "Return only valid JSON. Do not include markdown. "
        "Separate observable evidence from role-based inference."
    )
    user = f"""
Task: assign labels for a job-ad API excerpt.

Important evidence rules:
- The text is usually truncated to about 500 characters.
- For observed labels, use only evidence visible in the excerpt.
- Do not infer observed coding from employer prestige, title alone, or industry norms.
- Employer/product boilerplate does not count as role-specific AI signal.
- A company selling AI products does not mean the candidate's role uses AI.
- Physical data-centre, MEP, AutoCAD, network, sales, pre-sales, recruitment,
  support, finance, product-management and business-analysis work should not be
  treated as AI/coding unless the excerpt gives role-specific evidence.
- Use coding_likely_by_role for industry/common-sense inference.

Current labels for review:
{current}

Allowed ROLE values:
{_options(ROLE_VALUES)}

Allowed ENTRY_FIT values:
{_options(ENTRY_FIT_VALUES)}

Allowed AI_SIGNAL values:
{_options(AI_SIGNAL_VALUES)}

Allowed CODING_SIGNAL values:
{_options(CODING_SIGNAL_VALUES)}

Allowed coding_likely_by_role values:
{_options(CODING_LIKELY_VALUES)}

Return exactly this JSON object:
{{
  "observed": {{
    "role": "...",
    "entry_fit": "...",
    "ai_signal": "...",
    "coding_signal": "..."
  }},
  "inferred": {{
    "coding_likely_by_role": "..."
  }},
  "confidence": "high|medium|low",
  "rationale": "Short reason using the excerpt for observed labels; mention inference differences."
}}

Job excerpt:
<<<
{text}
>>>
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _as_object(payload: Any, key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"LLM response field '{key}' must be an object")
    return value


def _validate_choice(value: Any, allowed: set[str], field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"LLM response field '{field}' must be a string")
    if value not in allowed:
        raise ValueError(f"invalid {field}: {value}")
    return value


def parse_llm_audit_response(content: str) -> ParsedLLMAudit:
    """Parse and validate a JSON response from the LLM."""

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")

    observed = _as_object(payload, "observed")
    inferred = _as_object(payload, "inferred")
    confidence = _validate_choice(
        payload.get("confidence"), {"high", "medium", "low"}, "confidence"
    )
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("LLM response field 'rationale' must be a non-empty string")

    return ParsedLLMAudit(
        role=_validate_choice(observed.get("role"), ROLE_VALUES, "observed.role"),
        entry_fit=_validate_choice(
            observed.get("entry_fit"), ENTRY_FIT_VALUES, "observed.entry_fit"
        ),
        ai_signal=_validate_choice(
            observed.get("ai_signal"), AI_SIGNAL_VALUES, "observed.ai_signal"
        ),
        coding_signal=_validate_choice(
            observed.get("coding_signal"), CODING_SIGNAL_VALUES, "observed.coding_signal"
        ),
        coding_likely_by_role=_validate_choice(
            inferred.get("coding_likely_by_role"),
            CODING_LIKELY_VALUES,
            "inferred.coding_likely_by_role",
        ),
        confidence=confidence,
        rationale=rationale.strip(),
    )


def current_labels_by_dimension(labels: list[str]) -> dict[str, str]:
    """Return one label value per document-label dimension."""

    result: dict[str, str] = {}
    for label in labels:
        if "::" not in label:
            continue
        dimension, value = label.split("::", 1)
        if dimension in {"ROLE", "ENTRY_FIT", "AI_SIGNAL", "CODING_SIGNAL"}:
            result[dimension] = value
    return result


def disagreement_dimensions(current_labels: list[str], audit: ParsedLLMAudit) -> list[str]:
    """List observed-label dimensions where LLM and current labels disagree."""

    current = current_labels_by_dimension(current_labels)
    proposed = {
        "ROLE": audit.role,
        "ENTRY_FIT": audit.entry_fit,
        "AI_SIGNAL": audit.ai_signal,
        "CODING_SIGNAL": audit.coding_signal,
    }
    return [
        dimension
        for dimension, value in proposed.items()
        if current.get(dimension) != value
    ]


def summarize_llm_audit_review(rows: list[dict[str, str]]) -> dict[str, object]:
    """Summarize a private LLM-audit review CSV without source text."""

    total = len(rows)
    disagreement_rows = [
        row for row in rows if row.get("disagreement_dimensions", "").strip()
    ]
    dimension_counts: Counter[str] = Counter()
    combination_counts: Counter[str] = Counter()
    for row in disagreement_rows:
        dimensions = [
            value.strip()
            for value in row["disagreement_dimensions"].split("|")
            if value.strip()
        ]
        combination_counts[" + ".join(dimensions)] += 1
        dimension_counts.update(dimensions)

    confidence_counts = Counter(row.get("confidence", "") for row in rows)
    disagreement_confidence_counts = Counter(
        row.get("confidence", "") for row in disagreement_rows
    )
    coding_likely_counts = Counter(
        row.get("coding_likely_by_role", "") for row in rows
    )
    exact_agreement_share = (total - len(disagreement_rows)) / total if total else 0.0
    return {
        "rows": total,
        "rows_with_any_disagreement": len(disagreement_rows),
        "exact_agreement_share": round(exact_agreement_share, 4),
        "disagreement_by_dimension": dict(sorted(dimension_counts.items())),
        "top_disagreement_combinations": dict(combination_counts.most_common(10)),
        "confidence_distribution": dict(sorted(confidence_counts.items())),
        "disagreement_confidence_distribution": dict(
            sorted(disagreement_confidence_counts.items())
        ),
        "coding_likely_by_role_distribution": dict(sorted(coding_likely_counts.items())),
    }
