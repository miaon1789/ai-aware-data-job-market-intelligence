"""Transparent seniority signal extraction."""

from __future__ import annotations

import re

SENIOR_PATTERNS = (
    r"\bsenior\b",
    r"\blead\b",
    r"\bprincipal\b",
    r"\bmanager\b",
    r"\bhead of\b",
    r"\b[5-9]\+?\s+years?\b",
)
JUNIOR_PATTERNS = (
    r"\bgraduate\b",
    r"\bjunior\b",
    r"\bentry[- ]level\b",
    r"\bearly career\b",
    r"\b0\s*[-–]\s*2\s+years?\b",
    r"\bup to 2 years?\b",
)


def classify_seniority(title: str, description: str) -> tuple[str, str]:
    """Return a coarse label and the first matched evidence string."""

    text = f"{title}. {description}".lower()
    for pattern in SENIOR_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return "Non-Junior", match.group(0)
    for pattern in JUNIOR_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return "Graduate / Junior", match.group(0)
    return "Unclear", ""
