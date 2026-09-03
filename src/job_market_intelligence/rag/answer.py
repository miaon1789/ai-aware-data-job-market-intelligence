"""Grounded answer synthesis with citation checking.

Following the convention this repository already uses for LLM-assisted label
auditing, nothing is sent to a provider unless the caller passes an explicit
acknowledgement: the passages are licensed private advertisement text. The
default path builds the prompt and returns it unsent, which is enough to review
what would be transmitted.

Citations are verified against the passages that were actually supplied. A
model that cites an advertisement id it was not given has hallucinated the
provenance, and that is worth catching mechanically rather than trusting.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from .prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt


class AnswerValidationError(ValueError):
    """Raised when a model reply does not match the required answer schema."""


@dataclass(frozen=True)
class GroundedAnswer:
    answerable: bool
    answer: str
    citations: tuple[str, ...] = ()
    missing: str = ""
    unsupported_citations: tuple[str, ...] = ()
    prompt: str = ""
    sent: bool = False
    supporting_job_ids: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "answerable": self.answerable,
            "answer": self.answer,
            "citations": list(self.citations),
            "missing": self.missing,
            "unsupported_citations": list(self.unsupported_citations),
            "sent": self.sent,
            "supporting_job_ids": list(self.supporting_job_ids),
        }


def parse_answer(payload: str | dict[str, object]) -> dict[str, object]:
    """Strictly validate a model reply against the answer schema."""

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise AnswerValidationError(f"reply is not valid JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise AnswerValidationError("reply must be a JSON object")

    missing_keys = sorted({"answerable", "answer", "citations"} - set(payload))
    if missing_keys:
        raise AnswerValidationError(f"reply is missing keys: {', '.join(missing_keys)}")
    if not isinstance(payload["answerable"], bool):
        raise AnswerValidationError("answerable must be a boolean")
    if not isinstance(payload["answer"], str):
        raise AnswerValidationError("answer must be a string")
    if not isinstance(payload["citations"], list) or not all(
        isinstance(item, str) for item in payload["citations"]
    ):
        raise AnswerValidationError("citations must be a list of strings")
    return payload


def check_citations(
    citations: Sequence[str], passages: Sequence[dict[str, object]]
) -> tuple[str, ...]:
    """Return cited advertisement ids that were not among the passages."""

    supplied = {str(passage.get("job_id", "")) for passage in passages}
    return tuple(
        sorted({citation for citation in citations if citation.strip() not in supplied})
    )


def synthesise_answer(
    question: str,
    passages: Sequence[dict[str, object]],
    *,
    client=None,
    acknowledge_private_text: bool = False,
) -> GroundedAnswer:
    """Build -- and optionally send -- one grounded-answer request.

    ``client`` is any callable taking ``(system_prompt, user_prompt)`` and
    returning the model's raw reply. Leaving it unset, or omitting the
    acknowledgement, returns the assembled prompt without transmitting
    anything.
    """

    prompt = build_answer_prompt(question, passages)
    supporting = tuple(dict.fromkeys(str(passage.get("job_id", "")) for passage in passages))

    if not passages:
        return GroundedAnswer(
            answerable=False,
            answer="",
            missing="No passages were retrieved for this question.",
            prompt=prompt,
            sent=False,
            supporting_job_ids=supporting,
        )
    if client is None or not acknowledge_private_text:
        return GroundedAnswer(
            answerable=False,
            answer="",
            missing=(
                "Dry run: the prompt was assembled but not sent. Pass a client and "
                "acknowledge_private_text=True to send licensed advertisement text "
                "to the configured provider."
            ),
            prompt=prompt,
            sent=False,
            supporting_job_ids=supporting,
        )

    reply = parse_answer(client(ANSWER_SYSTEM_PROMPT, prompt))
    citations = tuple(str(item) for item in reply["citations"])
    return GroundedAnswer(
        answerable=bool(reply["answerable"]),
        answer=str(reply["answer"]),
        citations=citations,
        missing=str(reply.get("missing", "")),
        unsupported_citations=check_citations(citations, passages),
        prompt=prompt,
        sent=True,
        supporting_job_ids=supporting,
    )
