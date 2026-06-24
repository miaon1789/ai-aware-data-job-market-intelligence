"""Public benchmark evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from .skills import SkillExtractor


def load_skillspan(path: str | Path) -> list[dict[str, object]]:
    records = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        lengths = {
            len(record.get("tokens", [])),
            len(record.get("tags_skill", [])),
            len(record.get("tags_knowledge", [])),
        }
        if len(lengths) != 1:
            raise ValueError(f"SkillSpan line {line_number} has inconsistent tag lengths")
        records.append(record)
    return records


def _metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, object]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + fn + tn
    return {
        "samples": total,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
    }


def evaluate_skillspan_sentence_detection(
    records: list[dict[str, object]], extractor: SkillExtractor | None = None
) -> dict[str, object]:
    """Evaluate whether each sentence contains any annotated skill/knowledge."""

    active_extractor = extractor or SkillExtractor()
    counts: dict[str, list[int]] = {"overall": [0, 0, 0, 0]}
    for record in records:
        source = str(record.get("source", "unknown"))
        counts.setdefault(source, [0, 0, 0, 0])
        gold = any(tag != "O" for tag in record["tags_skill"]) or any(
            tag != "O" for tag in record["tags_knowledge"]
        )
        text = " ".join(str(token) for token in record["tokens"])
        predicted = bool(active_extractor.extract(text))
        position = 0 if gold and predicted else 1 if predicted else 2 if gold else 3
        counts["overall"][position] += 1
        counts[source][position] += 1

    return {name: _metrics(*values) for name, values in sorted(counts.items())}
