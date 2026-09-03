"""Validated records shared by ingestion and modelling."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .locations import SUPPORTED_CITIES, normalise_au_city

RoleLabel = Literal[
    "Data Analyst",
    "Data Analyst / BI",
    "Data Scientist",
    "Data Scientist / ML",
    "Data Engineer",
    "ML/AI Engineer",
    "AI / Automation",
    "Other / Mixed",
]


class JobAd(BaseModel):
    """Canonical job-advertisement schema for the MVP."""

    model_config = ConfigDict(str_strip_whitespace=True)

    job_id: str = Field(min_length=1)
    title: str = Field(min_length=2)
    description: str = Field(min_length=20)
    city: str
    company: str = Field(min_length=1)
    posted_at: date
    source: str = Field(min_length=1)
    source_url: str = ""
    role_label: RoleLabel | None = None

    @field_validator("city")
    @classmethod
    def validate_city(cls, value: str) -> str:
        canonical = normalise_au_city(value)
        if canonical is None:
            supported = ", ".join(sorted(SUPPORTED_CITIES))
            raise ValueError(f"city must be an Australian location (one of: {supported})")
        return canonical


class SkillDefinition(BaseModel):
    name: str
    category: str
    aliases: list[str] = Field(min_length=1)


class SkillMention(BaseModel):
    job_id: str
    skill: str
    category: str
    context: Literal["required", "preferred", "negated", "alternative", "mentioned"]
    sentence: str
