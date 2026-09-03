"""Deciding which tool a question belongs to.

A retrieval system asked "how many advertisements require SQL?" will return
its five best-matching passages and an answer synthesised from them will state
a number. The number will be wrong, and it will be stated with the same
confidence as a correct one, because nothing in the pipeline distinguishes
"the five most relevant advertisements" from "all the advertisements". Counting
needs the whole population; retrieval is built to discard most of it.

This module classifies a question before it is answered, so that population
questions go to SQL over the analysis database and only descriptive questions
go to retrieval. The same taxonomy labels the evaluation set, which means the
classifier can be measured against it rather than assumed correct.
"""

from __future__ import annotations

import re
from typing import Literal

QueryClass = Literal["lexical", "semantic", "aggregation"]
ToolName = Literal["search_job_ads", "query_job_stats"]

# Population questions: the answer is a count, share, ranking or comparison
# over every advertisement, not a passage from one of them.
AGGREGATION_PATTERNS = (
    r"\bhow many\b",
    r"\bhow much\b",
    r"\bwhat (?:share|proportion|percentage|fraction)\b",
    r"\bwhat is the (?:total|number|count|average|median|distribution|breakdown)\b",
    r"\b(?:count|tally|total)\b",
    r"\bmost common\b",
    r"\bleast common\b",
    r"\bwhich .{0,40}\b(?:most|least|more|fewer|highest|lowest)\b",
    r"\bcompare\b",
    r"\bdistribution\b",
    r"\bbreakdown\b",
    r"\bper cent\b|\bpercent\b|\b%\b",
    r"\bon average\b",
    r"\btrend\b",
    r"\bthe number of\b",
    r"\bare there (?:more|fewer|any)\b",
    r"\brank\b",
    r"\btop (?:\d+|three|five|ten)\b",
    r"\bhow often\b",
    r"\bmore often\b|\bmentioned more\b",
    r"\bacross the (?:whole|entire|full)\b",
    r"\bevery advertisement\b|\ball advertisements\b",
)

# A lexical query names a thing. It is short and carries no clause structure.
MAX_LEXICAL_WORDS = 4
SEMANTIC_MARKERS = (
    r"\broles?\b",
    r"\bjobs?\b",
    r"\bpositions?\b",
    r"\badvertisements?\b",
    r"\bwork(?:ing)?\b",
    r"\bthat\b",
    r"\bwhere\b",
    r"\bwho\b",
    r"\binvolv",
    r"\bwould\b",
)

_AGGREGATION_RE = re.compile("|".join(AGGREGATION_PATTERNS), re.IGNORECASE)
_SEMANTIC_RE = re.compile("|".join(SEMANTIC_MARKERS), re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z0-9+#.]+")


def classify_query(question: str) -> QueryClass:
    """Label a question as lexical, semantic or aggregation."""

    text = str(question).strip()
    if not text:
        return "semantic"
    if _AGGREGATION_RE.search(text):
        return "aggregation"
    words = _WORD_RE.findall(text)
    if len(words) <= MAX_LEXICAL_WORDS and not _SEMANTIC_RE.search(text):
        return "lexical"
    return "semantic"


def route(question: str) -> ToolName:
    """Choose the tool that can actually answer ``question``."""

    return (
        "query_job_stats"
        if classify_query(question) == "aggregation"
        else "search_job_ads"
    )


def routing_rationale(question: str) -> str:
    """A short explanation, surfaced in API responses and tool output."""

    query_class = classify_query(question)
    if query_class == "aggregation":
        return (
            "Counting or comparing across the corpus requires every row, so this "
            "goes to SQL over the analysis database rather than to retrieval."
        )
    if query_class == "lexical":
        return (
            "A specific named tool or technology; lexical BM25 matching carries "
            "this query and dense retrieval is the weaker branch."
        )
    return (
        "A descriptive question about how advertisements are worded; retrieval "
        "over passages answers it, with dense matching carrying the query."
    )
