"""Golden evaluation set: hand-written queries with rule-derived relevance.

Writing sixty relevance judgements by hand would take most of a day and would
not be reproducible by anyone else. Instead each query carries a *rule* that
resolves to a document set against the private corpus, and it is the rule that
is committed to the repository rather than the resolved advertisement ids.
Three of the four rule types reuse work the project has already done and
validated: the 244-advertisement adjudicated label set, the dictionary skill
extractor, and the cleaned metadata.

Known bias, stated here because it shapes every number downstream: queries
derived from an existing labelling scheme can only test the dimensions that
scheme covers. See ``eval/golden/README.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

# "mixed" queries carry a lexical constraint *and* a semantic one in the same
# sentence ("Python roles in Melbourne that involve explaining results to
# non-technical people"). They exist to test the case single-branch routing is
# least equipped for and fusion was invented for.
QUERY_CLASSES = ("lexical", "semantic", "aggregation", "mixed")
RULE_TYPES = ("alias_match", "skill_match", "label_match", "metadata_match", "all_of")
JUDGED_POOLS = ("all", "labelled")


def document_hash(job_id: str) -> str:
    """Stable, non-resolvable document identifier for published artefacts.

    Matches the 20-character convention already used by the annotation
    manifest, so published evaluation files never carry a job id that resolves
    against the private corpus or against the source site.

    This is a de-identifier, not a one-way function over a secret. The digest
    is unsalted and advertisement ids are short numeric strings, so anyone
    holding the id space can recompute the mapping. That is an accepted trade:
    the file stays reproducible by a reader who has their own corpus, and the
    most an attacker recovers is which public listing matched a public query.
    Salting would buy secrecy the published artefacts do not need and would
    cost the reproducibility they exist for.
    """

    return hashlib.sha256(str(job_id).encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class GoldenQuery:
    query_id: str
    query: str
    query_class: str
    relevance: Mapping[str, object]
    judged_pool: str = "all"
    stats_tool: Mapping[str, object] | None = None
    outside_skill_dictionary: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.query_class not in QUERY_CLASSES:
            raise ValueError(f"query_class must be one of: {', '.join(QUERY_CLASSES)}")
        if self.judged_pool not in JUDGED_POOLS:
            raise ValueError(f"judged_pool must be one of: {', '.join(JUDGED_POOLS)}")
        rule_type = self.relevance.get("type")
        if rule_type not in RULE_TYPES:
            raise ValueError(f"relevance.type must be one of: {', '.join(RULE_TYPES)}")


@dataclass
class GoldenSet:
    queries: list[GoldenQuery] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)

    def by_class(self, query_class: str) -> list[GoldenQuery]:
        return [query for query in self.queries if query.query_class == query_class]


def load_queries(path: str | Path) -> GoldenSet:
    """Read queries.jsonl, rejecting duplicate ids and unknown rule types."""

    queries: list[GoldenQuery] = []
    seen: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from None
        query = GoldenQuery(
            query_id=str(payload["query_id"]),
            query=str(payload["query"]),
            query_class=str(payload["query_class"]),
            relevance=payload["relevance"],
            judged_pool=str(payload.get("judged_pool", "all")),
            stats_tool=payload.get("stats_tool"),
            outside_skill_dictionary=bool(payload.get("outside_skill_dictionary", False)),
            notes=str(payload.get("notes", "")),
        )
        if query.query_id in seen:
            raise ValueError(f"duplicate query_id: {query.query_id}")
        seen.add(query.query_id)
        queries.append(query)
    return GoldenSet(queries)


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(str(alias).lower()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def _searchable_text(jobs: pd.DataFrame) -> pd.Series:
    structured = jobs.get("description_structured")
    body = jobs["description"] if structured is None else structured.where(
        structured.astype(str).str.len() > 0, jobs["description"]
    )
    return (jobs["title"].astype(str) + "\n" + body.astype(str)).str.lower()


def resolve_relevance(
    query: GoldenQuery,
    *,
    jobs: pd.DataFrame,
    skill_mentions: pd.DataFrame | None = None,
    document_labels: pd.DataFrame | None = None,
) -> set[str]:
    """Resolve one query's rule to the set of relevant advertisement ids."""

    rule = query.relevance
    rule_type = str(rule["type"])

    if rule_type == "all_of":
        # An advertisement is relevant only if it satisfies every clause, which
        # is what makes a mixed query genuinely harder than either half.
        parts = [
            resolve_relevance(
                replace(query, relevance=sub_rule),
                jobs=jobs,
                skill_mentions=skill_mentions,
                document_labels=document_labels,
            )
            for sub_rule in rule["rules"]
        ]
        return set.intersection(*parts) if parts else set()

    if rule_type == "alias_match":
        text = _searchable_text(jobs)
        patterns = [_alias_pattern(alias) for alias in rule["aliases"]]
        mask = text.map(lambda value: any(pattern.search(value) for pattern in patterns))
        return set(jobs.loc[mask, "job_id"].astype(str))

    if rule_type == "skill_match":
        if skill_mentions is None:
            raise ValueError("skill_match rules require skill_mentions")
        wanted = set(rule["skills"])
        matched = skill_mentions[skill_mentions["skill"].isin(wanted)]
        if "contexts" in rule:
            matched = matched[matched["context"].isin(set(rule["contexts"]))]
        known = set(jobs["job_id"].astype(str))
        return set(matched["job_id"].astype(str)) & known

    if rule_type == "label_match":
        if document_labels is None:
            raise ValueError("label_match rules require document_labels")
        column = str(rule["column"])
        if column not in document_labels.columns:
            raise ValueError(f"document_labels has no column {column!r}")
        matched = document_labels[document_labels[column].isin(set(rule["values"]))]
        known = set(jobs["job_id"].astype(str))
        return set(matched["job_id"].astype(str)) & known

    column = str(rule["column"])
    if column not in jobs.columns:
        raise ValueError(f"jobs has no column {column!r}")
    matched = jobs[jobs[column].astype(str).isin(set(rule["values"]))]
    return set(matched["job_id"].astype(str))


def judged_documents(
    query: GoldenQuery,
    *,
    jobs: pd.DataFrame,
    document_labels: pd.DataFrame | None = None,
) -> set[str] | None:
    """Return the documents this query has judgements for, or None for all.

    Label-derived queries can only judge the 244 hand-labelled advertisements.
    Scoring them against the whole corpus would count an unjudged advertisement
    as a miss and would make semantic queries look systematically worse than
    lexical ones for a reason that has nothing to do with retrieval. Restricting
    to the judged pool -- evaluating over a condensed ranking -- keeps the two
    query classes comparable, at the cost of measuring semantic queries on a
    smaller corpus. Both facts are reported.
    """

    if query.judged_pool == "all":
        return None
    if document_labels is None:
        raise ValueError("a labelled judged_pool requires document_labels")
    known = set(jobs["job_id"].astype(str))
    return set(document_labels["job_id"].astype(str)) & known


def condense(ranked_documents: list[str], judged: set[str] | None) -> list[str]:
    """Drop unjudged documents from a ranking, preserving order."""

    if judged is None:
        return list(ranked_documents)
    return [document for document in ranked_documents if document in judged]
