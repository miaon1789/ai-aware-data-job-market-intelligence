"""Text normalisation and duplicate detection."""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher

import pandas as pd

WHITESPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^a-z0-9+#.]+")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
AU_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?61[\s()-]?|0)[2-478](?:[\s()-]?\d){8}(?!\d)"
)


def clean_text(value: str) -> str:
    """Collapse whitespace while preserving human-readable punctuation."""

    return WHITESPACE_RE.sub(" ", str(value)).strip()


def canonical_text(value: str) -> str:
    """Create a conservative comparison form for duplicate detection."""

    lowered = clean_text(value).lower()
    return WHITESPACE_RE.sub(" ", NON_WORD_RE.sub(" ", lowered)).strip()


def redact_personal_contacts(value: str) -> str:
    """Remove common recruiter email and Australian phone-number patterns."""

    redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", str(value))
    return AU_PHONE_RE.sub("[REDACTED_PHONE]", redacted)


def _fingerprint(row: pd.Series) -> str:
    content = "|".join(
        canonical_text(str(row[column]))
        for column in ("title", "company", "city", "description")
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:20]


def deduplicate_job_ads(
    frame: pd.DataFrame, near_duplicate_threshold: float = 0.94
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove exact and conservative near duplicates.

    Near-duplicate comparisons are limited to rows with the same canonical
    company, city and title. The newest record is retained.
    """

    if not 0.0 <= near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")

    working = frame.copy()
    for column in ("title", "description", "company"):
        working[column] = working[column].map(clean_text)
    working["description"] = working["description"].map(redact_personal_contacts)
    working["posted_at"] = pd.to_datetime(working["posted_at"]).dt.date.astype(str)
    working["duplicate_group"] = working.apply(_fingerprint, axis=1)
    working = working.sort_values(["posted_at", "job_id"], ascending=[False, True])

    kept_indexes: list[int] = []
    removed: list[dict[str, str | float]] = []
    representatives: dict[tuple[str, str, str], list[int]] = {}

    for index, row in working.iterrows():
        key = (
            canonical_text(row["company"]),
            canonical_text(row["city"]),
            canonical_text(row["title"]),
        )
        matched_index: int | None = None
        similarity = 0.0
        description = canonical_text(row["description"])
        for candidate_index in representatives.get(key, []):
            candidate = canonical_text(working.at[candidate_index, "description"])
            score = SequenceMatcher(None, description, candidate).ratio()
            if score >= near_duplicate_threshold:
                matched_index = candidate_index
                similarity = score
                break

        if matched_index is None:
            kept_indexes.append(index)
            representatives.setdefault(key, []).append(index)
        else:
            working.at[index, "duplicate_group"] = working.at[matched_index, "duplicate_group"]
            removed.append(
                {
                    "removed_job_id": str(row["job_id"]),
                    "kept_job_id": str(working.at[matched_index, "job_id"]),
                    "similarity": round(similarity, 4),
                }
            )

    kept = working.loc[kept_indexes].sort_values("job_id").reset_index(drop=True)
    audit = pd.DataFrame(removed, columns=["removed_job_id", "kept_job_id", "similarity"])
    return kept, audit
