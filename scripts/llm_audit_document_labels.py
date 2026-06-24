#!/usr/bin/env python3
"""Run or preview LLM-assisted audits for private document labels."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_market_intelligence.annotation import task_hash  # noqa: E402
from job_market_intelligence.document_label_suggestions import title_from_text  # noqa: E402
from job_market_intelligence.llm_annotation import (  # noqa: E402
    build_llm_audit_messages,
    disagreement_dimensions,
    llm_audit_json_schema,
    parse_llm_audit_response,
)

PRIVATE_ROOT = (ROOT / "data/private").resolve()


def inside_private(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == PRIVATE_ROOT or PRIVATE_ROOT in resolved.parents


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    allowed = {
        "LLM_API_BASE",
        "LLM_API_KEY",
        "LLM_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        if name in allowed:
            os.environ.setdefault(name, value.strip().strip("\"'"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def call_chat_completions(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: float,
) -> str:
    endpoint = api_base.rstrip("/") + "/chat/completions"
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if "anthropic.com" in api_base.lower():
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": llm_audit_json_schema(),
        }
    else:
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "JobMarketIntelligenceResearch/0.1 private-llm-audit",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM API returned HTTP {exc.code}: {detail}") from None
    except URLError as exc:
        raise RuntimeError(f"LLM API could not be reached: {exc.reason}") from None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM API response is missing choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("LLM API response is missing message.content")
    return message["content"]


def write_prompt_preview(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--document-export",
        type=Path,
        default=ROOT
        / "data/private/annotations_expanded/exports/"
        / "documents-annotator-1.business_context_v3_candidate.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data/private/annotations_expanded/exports/documents-llm-audit.jsonl",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=ROOT / "data/private/annotations_expanded/llm_audit_review.csv",
    )
    parser.add_argument(
        "--prompt-preview",
        type=Path,
        default=ROOT / "data/private/annotations_expanded/llm_prompt_preview.json",
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument(
        "--acknowledge-private-text",
        action="store_true",
        help="Required with --run-api because private job excerpts are sent to the LLM API.",
    )
    args = parser.parse_args()

    for path in [args.document_export, args.output, args.review_csv, args.prompt_preview]:
        if not inside_private(path):
            parser.error("LLM audit inputs and outputs must stay under data/private")
    if args.limit < 1 or args.offset < 0:
        parser.error("--limit must be positive and --offset cannot be negative")
    if not args.document_export.is_file():
        parser.error(f"document export does not exist: {args.document_export}")
    if args.run_api and not args.acknowledge_private_text:
        parser.error("--run-api requires --acknowledge-private-text")

    rows = read_jsonl(args.document_export)[args.offset : args.offset + args.limit]
    if not rows:
        parser.error("selected range contains no rows")

    prepared = []
    for row in rows:
        text = row.get("text")
        labels = row.get("label", row.get("labels"))
        if not isinstance(text, str) or not isinstance(labels, list):
            parser.error("each document row must contain text and label list")
        prepared.append(
            {
                "task_hash": task_hash(text),
                "title": title_from_text(text),
                "messages": build_llm_audit_messages(text, current_labels=[str(x) for x in labels]),
            }
        )

    if not args.run_api:
        write_prompt_preview(args.prompt_preview, prepared)
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "selected_rows": len(prepared),
                    "prompt_preview": str(args.prompt_preview),
                    "next_step": (
                        "rerun with --run-api --acknowledge-private-text after "
                        "setting LLM_API_KEY and LLM_MODEL"
                    ),
                },
                indent=2,
            )
        )
        return

    load_env_file(args.env_file)
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    api_base = (
        os.environ.get("LLM_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    model = os.environ.get("LLM_MODEL", "")
    if not api_key:
        parser.error("set LLM_API_KEY or OPENAI_API_KEY in the environment or .env")
    if not model:
        parser.error("set LLM_MODEL in the environment or .env")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    review_rows = []
    audit_rows = []
    for index, row in enumerate(rows):
        text = str(row["text"])
        labels = [str(x) for x in row.get("label", row.get("labels", []))]
        messages = build_llm_audit_messages(text, current_labels=labels)
        content = call_chat_completions(
            api_base=api_base,
            api_key=api_key,
            model=model,
            messages=messages,
            timeout=args.timeout,
        )
        audit = parse_llm_audit_response(content)
        disagreements = disagreement_dimensions(labels, audit)
        title = title_from_text(text)
        audit_row = {
            "id": row.get("id"),
            "text": text,
            "label": audit.observed_labels,
            "Comments": [],
            "task_hash": task_hash(text),
            "title": title,
            "current_label": labels,
            "llm_observed_label": audit.observed_labels,
            "coding_likely_by_role": audit.coding_likely_by_role,
            "confidence": audit.confidence,
            "rationale": audit.rationale,
            "disagreement_dimensions": disagreements,
            "username": "llm-audit",
        }
        audit_rows.append(audit_row)
        review_rows.append(
            {
                "task_hash": audit_row["task_hash"],
                "title": title,
                "current_labels": " | ".join(labels),
                "llm_observed_labels": " | ".join(audit.observed_labels),
                "coding_likely_by_role": audit.coding_likely_by_role,
                "confidence": audit.confidence,
                "disagreement_dimensions": " | ".join(disagreements),
                "rationale": audit.rationale,
            }
        )
        if args.request_delay_seconds > 0 and index < len(rows) - 1:
            time.sleep(args.request_delay_seconds)

    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows),
        encoding="utf-8",
    )
    args.output.chmod(0o600)
    with args.review_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "task_hash",
                "title",
                "current_labels",
                "llm_observed_labels",
                "coding_likely_by_role",
                "confidence",
                "disagreement_dimensions",
                "rationale",
            ],
        )
        writer.writeheader()
        writer.writerows(review_rows)
    args.review_csv.chmod(0o600)
    print(
        json.dumps(
            {
                "mode": "run-api",
                "audited_rows": len(audit_rows),
                "output": str(args.output),
                "review_csv": str(args.review_csv),
                "policy": "LLM audit output is a review aid; human adjudication remains required.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
