import json

import pytest

from job_market_intelligence.benchmarks import (
    evaluate_skillspan_sentence_detection,
    load_skillspan,
)


def record(tokens, skill_tags, knowledge_tags, source="tech"):
    return {
        "tokens": tokens,
        "tags_skill": skill_tags,
        "tags_knowledge": knowledge_tags,
        "source": source,
    }


def test_sentence_level_skillspan_metrics():
    records = [
        record(["Python", "required"], ["B", "O"], ["O", "O"]),
        record(["Friendly", "team"], ["O", "O"], ["O", "O"], source="house"),
        record(["UnknownTool"], ["B"], ["O"]),
    ]
    metrics = evaluate_skillspan_sentence_detection(records)
    assert metrics["overall"]["true_positive"] == 1
    assert metrics["overall"]["false_negative"] == 1
    assert metrics["overall"]["true_negative"] == 1
    assert metrics["overall"]["precision"] == 1.0
    assert metrics["overall"]["recall"] == 0.5


def test_skillspan_loader_validates_tag_lengths(tmp_path):
    path = tmp_path / "test.json"
    path.write_text(
        json.dumps(record(["Python"], ["B"], ["O"])) + "\n", encoding="utf-8"
    )
    assert len(load_skillspan(path)) == 1
    path.write_text(
        json.dumps(record(["Python"], ["B", "I"], ["O"])) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inconsistent"):
        load_skillspan(path)
