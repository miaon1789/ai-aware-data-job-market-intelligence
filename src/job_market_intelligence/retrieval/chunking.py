"""Structure-aware chunking of job advertisements.

Job advertisements are not free-flowing prose. They are a sequence of labelled
sections -- what the company does, what the role involves, what the candidate
must bring, what the package is -- and the requirement sections are the ones
that answer most retrieval queries. Splitting on a fixed character count cuts
requirement lists in half and strands bullets from the heading that gives them
meaning, so this module splits on the real section boundaries first and only
then packs sentences and bullets into token-budgeted windows.

Advertisements collected from an API that truncates the description (the
Adzuna free tier caps it near 500 characters) carry no section structure at
all. They fall through to a single ``intro`` section and produce one chunk at
every window size, which is a property of the source rather than a bug.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# Average English tokenisation runs close to four characters per token. The
# chunker uses this approximation rather than importing a tokeniser so that
# chunk boundaries stay deterministic and independent of the embedding model
# being ablated. Window sizes are therefore nominal, not exact token counts.
CHARS_PER_TOKEN = 4

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
BULLET_PREFIX_RE = re.compile(r"^[-*•‣◦⁃∙]\s+")
MAX_HEADING_CHARS = 90

# Canonical section names, ordered so that the most specific pattern wins.
SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "responsibilities",
        re.compile(
            r"responsibilit|what you.{0,4}ll (?:do|be doing)|what you will do|your impact"
            r"|day[- ]to[- ]day|key duties|the job|in this role|what the role involves"
            r"|your role|core duties|accountabilities|^you will|the impact you"
            r"|what you.{0,4}ll be working on|your mission",
            re.IGNORECASE,
        ),
    ),
    (
        "preferred",
        re.compile(
            r"nice[- ]to[- ]have|bonus points|preferred qualification|desirable"
            r"|great if you|even better|advantageous|highly regarded",
            re.IGNORECASE,
        ),
    ),
    (
        "requirements",
        re.compile(
            r"requirement|qualification|what you.{0,4}ll bring|what you bring"
            r"|what we.{0,4}re looking for|who you are|about you|your (?:skills|experience)"
            r"|skills (?:and|&) experience|must have|essential|you have|we.{0,4}re looking for"
            r"|minimum qualification|ideal candidate|your background|what we look for"
            r"|attributes we value|what we value|skills we|experience we",
            re.IGNORECASE,
        ),
    ),
    (
        "benefits",
        re.compile(
            r"benefit|perks|what we offer|why join|why work|compensation|salary|remuneration"
            r"|our offer|what.{0,4}s in it for you|life at|we offer",
            re.IGNORECASE,
        ),
    ),
    (
        "process",
        re.compile(
            r"how to apply|application process|interview process|next steps|hiring process"
            r"|recruitment process|to apply",
            re.IGNORECASE,
        ),
    ),
    (
        # Boilerplate every large employer appends: equal-opportunity statements,
        # privacy notices, recruiter-fraud warnings, pay-transparency clauses.
        # It is worth labelling precisely because it is worth filtering out.
        "legal",
        re.compile(
            r"equal (?:opportunity|employment)|diversity|inclusion|belonging|accessibility"
            r"|privacy|we are committed to|affirmative action|reasonable adjustment"
            r"|applicant safety|fraud|third[- ]party recruiter|compliance|pay transparency"
            r"|accommodation|e-verify|background check",
            re.IGNORECASE,
        ),
    ),
    (
        "role",
        re.compile(
            r"about the role|the role|role overview|position summary|the opportunity"
            r"|job description|the position|role purpose|purpose of the role"
            r"|what.{0,4}s unique about|why this role",
            re.IGNORECASE,
        ),
    ),
    (
        "about",
        re.compile(
            r"^about\b|who we are|our story|company overview|our growth"
            r"|our mission|the company|our team|introduction|life at",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class Section:
    """A heading and the body text that follows it."""

    name: str
    heading: str
    body: str


@dataclass(frozen=True)
class Chunk:
    """One retrievable window of a single advertisement."""

    chunk_id: str
    job_id: str
    chunk_index: int
    section: str
    heading: str
    body: str
    text: str
    token_estimate: int


def estimate_tokens(text: str) -> int:
    """Approximate the token length of ``text`` (see ``CHARS_PER_TOKEN``)."""

    return max(1, round(len(str(text)) / CHARS_PER_TOKEN))


MAX_HEADING_WORDS = 12


def _is_heading_shaped(line: str) -> bool:
    """Detect a line positioned as a heading rather than as body prose.

    Keyword matching alone is not enough: a requirement bullet such as
    "minimum of 15 hours a week" contains a section keyword but is body text.
    A heading opens with a capital, stays short, and does not close like a
    sentence.
    """

    stripped = line.strip()
    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return False
    if BULLET_PREFIX_RE.match(stripped):
        return False
    if len(stripped.split()) > MAX_HEADING_WORDS:
        return False
    if stripped.endswith(":"):
        return True
    if not stripped[0].isupper() and not stripped[0].isdigit():
        return False
    return not stripped.endswith((".", ",", ";"))


def classify_heading(line: str) -> str | None:
    """Return the canonical section a heading line introduces, if it is one."""

    if not _is_heading_shaped(line):
        return None
    stripped = line.strip().rstrip(":").strip()
    for name, pattern in SECTION_PATTERNS:
        if pattern.search(stripped):
            return name
    return None


def detect_sections(structured_text: str) -> list[Section]:
    """Split newline-delimited advertisement text into labelled sections.

    Text with no recognisable headings -- truncated API excerpts, most often --
    returns a single ``intro`` section holding the whole description.
    """

    lines = [line for line in str(structured_text).split("\n")]
    sections: list[Section] = []
    current_name = "intro"
    current_heading = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body or current_heading:
            sections.append(Section(name=current_name, heading=current_heading, body=body))

    for line in lines:
        if not line.strip():
            buffer.append("")
            continue
        name = classify_heading(line)
        if name is None and _is_heading_shaped(line) and buffer:
            name = "other"
        if name is not None:
            flush()
            buffer = []
            current_name = name
            current_heading = line.strip().rstrip(":").strip()
            continue
        buffer.append(line)

    flush()
    return [section for section in sections if section.body or section.heading]


def split_units(body: str) -> list[str]:
    """Split section text into indivisible units: whole bullets or sentences."""

    units: list[str] = []
    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if BULLET_PREFIX_RE.match(stripped):
            units.append(stripped)
            continue
        units.extend(part.strip() for part in SENTENCE_SPLIT_RE.split(stripped) if part.strip())
    return units


def _hard_split(unit: str, target_tokens: int) -> list[str]:
    """Break a single oversized unit on word boundaries as a last resort."""

    budget = target_tokens * CHARS_PER_TOKEN
    words = unit.split()
    pieces: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if current and length + len(word) + 1 > budget:
            pieces.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + 1
    if current:
        pieces.append(" ".join(current))
    return pieces or [unit]


def _pack_units(
    units: list[str], *, target_tokens: int, overlap_ratio: float
) -> list[str]:
    """Greedily pack units into windows, repeating trailing units as overlap."""

    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        for piece in (
            [unit] if estimate_tokens(unit) <= target_tokens else _hard_split(unit, target_tokens)
        ):
            piece_tokens = estimate_tokens(piece)
            if current and current_tokens + piece_tokens > target_tokens:
                windows.append("\n".join(current))
                carry: list[str] = []
                carry_budget = target_tokens * overlap_ratio
                for previous in reversed(current):
                    if estimate_tokens("\n".join(carry)) >= carry_budget:
                        break
                    carry.insert(0, previous)
                # An overlap that swallows the whole window would never advance.
                if len(carry) >= len(current):
                    carry = carry[-1:] if current else []
                current = list(carry)
                current_tokens = estimate_tokens("\n".join(current)) if current else 0
            current.append(piece)
            current_tokens += piece_tokens

    if current:
        windows.append("\n".join(current))
    return windows


def chunk_advertisement(
    structured_text: str,
    *,
    job_id: str,
    title: str = "",
    target_tokens: int = 512,
    overlap_ratio: float = 0.1,
    min_chunk_tokens: int = 48,
    include_title: bool = True,
) -> list[Chunk]:
    """Split one advertisement into retrievable chunks.

    Each chunk is indexed together with the advertisement title and its section
    heading. A chunk lifted from the middle of a long advertisement otherwise
    carries no indication of which role it belongs to, which hurts both lexical
    and dense retrieval; the unprefixed ``body`` is kept for citation display.
    """

    if target_tokens < 32:
        raise ValueError("target_tokens must be at least 32")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1)")

    pending: list[tuple[str, str, str]] = []
    for section in detect_sections(structured_text):
        units = split_units(section.body)
        if not units:
            if section.heading:
                pending.append((section.name, section.heading, ""))
            continue
        for window in _pack_units(
            units, target_tokens=target_tokens, overlap_ratio=overlap_ratio
        ):
            pending.append((section.name, section.heading, window))

    merged = _merge_small_windows(pending, min_chunk_tokens=min_chunk_tokens, cap=target_tokens)

    chunks: list[Chunk] = []
    for index, (name, heading, body) in enumerate(merged):
        prefix_parts = [part for part in (title if include_title else "", heading) if part]
        text = "\n".join([*prefix_parts, body]).strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{job_id}::{index:02d}",
                job_id=str(job_id),
                chunk_index=index,
                section=name,
                heading=heading,
                body=body,
                text=text,
                token_estimate=estimate_tokens(text),
            )
        )
    return chunks


def _merge_small_windows(
    windows: list[tuple[str, str, str]], *, min_chunk_tokens: int, cap: int
) -> list[tuple[str, str, str]]:
    """Fold undersized *continuation* fragments back into their own section.

    Packing a long section can leave a stub as its final window -- two lines of
    a requirement list stranded from the rest. Those retrieve poorly, so they
    are merged back while the combined window still fits the budget.

    Merging stops at a section boundary. A short section is left as its own
    chunk even when it is tiny: folding "Nice to have" into the preceding
    responsibilities window would mislabel it and blur exactly the distinction
    that structure-aware chunking exists to preserve.
    """

    merged: list[tuple[str, str, str]] = []
    for name, heading, body in windows:
        if not body:
            # A heading with no body of its own carries no retrievable content.
            continue
        if (
            merged
            and merged[-1][0] == name
            and merged[-1][1] == heading
            and estimate_tokens(body) < min_chunk_tokens
            and estimate_tokens(merged[-1][2]) + estimate_tokens(body) <= cap
        ):
            previous_name, previous_heading, previous_body = merged[-1]
            merged[-1] = (previous_name, previous_heading, f"{previous_body}\n{body}")
            continue
        merged.append((name, heading, body))
    return merged


CHUNK_METADATA_COLUMNS = (
    "city",
    "company",
    "posted_at",
    "source",
    "analysis_role",
    "seniority",
)


def build_chunk_frame(
    jobs: pd.DataFrame,
    *,
    target_tokens: int = 512,
    overlap_ratio: float = 0.1,
    min_chunk_tokens: int = 48,
    include_title: bool = True,
) -> pd.DataFrame:
    """Chunk every advertisement and attach the metadata used for filtering."""

    required = {"job_id", "title", "description"}
    missing = sorted(required - set(jobs.columns))
    if missing:
        raise ValueError(f"jobs are missing columns: {', '.join(missing)}")

    records: list[dict[str, object]] = []
    for row in jobs.to_dict(orient="records"):
        structured = str(row.get("description_structured") or "").strip()
        text = structured or str(row["description"])
        chunks = chunk_advertisement(
            text,
            job_id=str(row["job_id"]),
            title=str(row.get("title", "")),
            target_tokens=target_tokens,
            overlap_ratio=overlap_ratio,
            min_chunk_tokens=min_chunk_tokens,
            include_title=include_title,
        )
        for chunk in chunks:
            record: dict[str, object] = {
                "chunk_id": chunk.chunk_id,
                "job_id": chunk.job_id,
                "chunk_index": chunk.chunk_index,
                "section": chunk.section,
                "heading": chunk.heading,
                "body": chunk.body,
                "text": chunk.text,
                "token_estimate": chunk.token_estimate,
                "title": str(row.get("title", "")),
                # Truncated sources cannot be chunked meaningfully; the flag lets
                # every evaluation report separate them from full-text ads.
                "is_truncated": not structured or structured == str(row["description"]),
            }
            for column in CHUNK_METADATA_COLUMNS:
                record[column] = str(row.get(column, ""))
            records.append(record)

    columns = [
        "chunk_id",
        "job_id",
        "chunk_index",
        "section",
        "heading",
        "body",
        "text",
        "token_estimate",
        "title",
        "is_truncated",
        *CHUNK_METADATA_COLUMNS,
    ]
    return pd.DataFrame(records, columns=columns)
