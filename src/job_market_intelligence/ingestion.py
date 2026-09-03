"""CSV ingestion with row-level validation errors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from .schema import JobAd

REQUIRED_COLUMNS = {
    "job_id",
    "title",
    "description",
    "city",
    "company",
    "posted_at",
    "source",
    "source_url",
    "role_label",
}
PRIVATE_PROVENANCE_COLUMNS = ("collection_query", "collection_group", "retrieved_at")
# Full-text sources also carry a newline-delimited copy of the description for
# structure-aware chunking. It is optional so Adzuna-only inputs stay valid.
OPTIONAL_TEXT_COLUMNS = ("description_structured",)


class DataValidationError(ValueError):
    """Raised when an input dataset does not match the documented contract."""


def load_job_ads(path: str | Path) -> pd.DataFrame:
    """Load and validate job advertisements from a UTF-8 CSV file."""

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise DataValidationError(f"missing required columns: {', '.join(missing)}")

    records: list[dict[str, object]] = []
    errors: list[str] = []
    for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
        payload = {key: row[key] for key in REQUIRED_COLUMNS}
        if not payload["role_label"]:
            payload["role_label"] = None
        try:
            record = JobAd.model_validate(payload)
            validated = record.model_dump(mode="json")
            for column in PRIVATE_PROVENANCE_COLUMNS:
                validated[column] = str(row.get(column, "")).strip()
            for column in OPTIONAL_TEXT_COLUMNS:
                validated[column] = str(row.get(column, "")).strip() or validated["description"]
            records.append(validated)
        except ValidationError as exc:
            messages = "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in exc.errors()
            )
            errors.append(f"row {row_number}: {messages}")

    if errors:
        preview = "\n".join(errors[:10])
        suffix = "\n... additional errors omitted" if len(errors) > 10 else ""
        raise DataValidationError(f"invalid job data:\n{preview}{suffix}")

    return pd.DataFrame.from_records(records)
