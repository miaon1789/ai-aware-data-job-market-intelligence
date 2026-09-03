"""Prompts for grounded answer synthesis."""

from __future__ import annotations

from collections.abc import Sequence

ANSWER_SYSTEM_PROMPT = """You answer questions about a corpus of Australian job \
advertisements using only the numbered passages supplied to you.

Rules:
- Use only the supplied passages. Do not use outside knowledge about any \
company, role or technology.
- Cite the advertisement id for every claim, in square brackets, e.g. [gh-12345].
- If the passages do not contain enough information to answer, set "answerable" \
to false and say plainly what is missing. Do not guess.
- Never state a count, total, share or ranking over the whole corpus. The \
passages are a small retrieved sample, not the population; counting questions \
are answered by a separate SQL tool. If the question asks how many or what \
share, set "answerable" to false and say the question needs the statistics tool.
- Describe how the advertisements are worded. Do not infer what a job really \
requires beyond what the text says, and do not evaluate any candidate.

Reply with JSON only, matching the supplied schema."""


def answer_json_schema() -> dict[str, object]:
    """Structured-output schema, validated strictly on the way back in."""

    return {
        "name": "grounded_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answerable", "answer", "citations", "missing"],
            "properties": {
                "answerable": {"type": "boolean"},
                "answer": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
                "missing": {"type": "string"},
            },
        },
    }


def format_passages(passages: Sequence[dict[str, object]]) -> str:
    """Render retrieved chunks as a numbered, citable block."""

    lines: list[str] = []
    for index, passage in enumerate(passages, start=1):
        job_id = str(passage.get("job_id", ""))
        heading = str(passage.get("heading", "")).strip()
        title = str(passage.get("title", "")).strip()
        company = str(passage.get("company", "")).strip()
        header = f"[{index}] advertisement {job_id}"
        if title:
            header += f" — {title}"
        if company:
            header += f" at {company}"
        if heading:
            header += f" (section: {heading})"
        lines.append(f"{header}\n{str(passage.get('body', '')).strip()}")
    return "\n\n".join(lines)


def build_answer_prompt(question: str, passages: Sequence[dict[str, object]]) -> str:
    """Assemble the user message for one grounded-answer request."""

    return (
        f"Question: {question}\n\n"
        f"Passages:\n{format_passages(passages)}\n\n"
        "Answer using only these passages, citing advertisement ids."
    )
