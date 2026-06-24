"""Dictionary-based skill extraction with local context classification."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .schema import SkillDefinition, SkillMention

DEFAULT_SKILLS_PATH = Path(__file__).resolve().parents[2] / "config" / "skills.json"
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|[\r\n]+")

NEGATED_PATTERNS = (
    r"\bnot\s+(?:strictly\s+)?required\b",
    r"\bno\s+(?:prior\s+)?experience\s+(?:is\s+)?required\b",
    r"\bwithout\s+(?:the\s+need\s+for\s+)?experience\b",
)
PREFERRED_MARKERS = (
    "preferred",
    "desirable",
    "nice to have",
    "nice-to-have",
    "a bonus",
    "advantageous",
    "highly regarded",
)
REQUIRED_MARKERS = (
    "required",
    "essential",
    "must have",
    "must-have",
    "you will need",
    "proficiency in",
    "proficient in",
    "strong experience",
    "demonstrated experience",
)


def load_skill_definitions(path: str | Path = DEFAULT_SKILLS_PATH) -> list[SkillDefinition]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SkillDefinition.model_validate(item) for item in payload]


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


def classify_context(sentence: str, match_start: int, match_end: int) -> str:
    """Classify how a skill is presented within its sentence."""

    local = sentence[max(0, match_start - 100) : min(len(sentence), match_end + 100)].lower()
    if any(re.search(pattern, local) for pattern in NEGATED_PATTERNS):
        return "negated"
    if any(marker in local for marker in PREFERRED_MARKERS):
        return "preferred"
    if any(marker in local for marker in REQUIRED_MARKERS):
        return "required"

    after = sentence[match_end : min(len(sentence), match_end + 25)].lower()
    before = sentence[max(0, match_start - 25) : match_start].lower()
    if re.search(r"\bor\b", after) or re.search(r"\bor\b", before):
        return "alternative"
    return "mentioned"


class SkillExtractor:
    """Extract unique normalised skills from free text."""

    def __init__(self, definitions: list[SkillDefinition] | None = None) -> None:
        self.definitions = definitions or load_skill_definitions()
        self._patterns = [
            (definition, alias, _alias_pattern(alias))
            for definition in self.definitions
            for alias in sorted(definition.aliases, key=len, reverse=True)
        ]

    def extract(self, text: str, job_id: str = "ad-hoc") -> list[SkillMention]:
        mentions: dict[str, SkillMention] = {}
        priority = {"negated": 5, "required": 4, "preferred": 3, "alternative": 2, "mentioned": 1}

        for sentence in filter(None, (part.strip() for part in SENTENCE_SPLIT_RE.split(text))):
            for definition, _alias, pattern in self._patterns:
                match = pattern.search(sentence)
                if not match:
                    continue
                context = classify_context(sentence, match.start(), match.end())
                mention = SkillMention(
                    job_id=job_id,
                    skill=definition.name,
                    category=definition.category,
                    context=context,
                    sentence=sentence[:500],
                )
                existing = mentions.get(definition.name)
                if existing is None or priority[context] > priority[existing.context]:
                    mentions[definition.name] = mention

        return sorted(mentions.values(), key=lambda item: item.skill.lower())


def extract_mentions_for_jobs(frame, extractor: SkillExtractor | None = None):
    """Return one row per normalised skill and job."""

    import pandas as pd

    active_extractor = extractor or SkillExtractor()
    records = []
    for row in frame.itertuples(index=False):
        records.extend(
            mention.model_dump()
            for mention in active_extractor.extract(row.description, job_id=row.job_id)
        )
    return pd.DataFrame(
        records,
        columns=["job_id", "skill", "category", "context", "sentence"],
    )
