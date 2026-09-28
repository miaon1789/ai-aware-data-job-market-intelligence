---
name: validate-retrieval
description: Validate retrieval changes in this repository and review the resulting evidence. Use when the user asks to check retrieval changes, verify evaluation claims, or compare a candidate against an explicit baseline.
---

# Validate retrieval changes

Run the project's deterministic validator, then ask the read-only
`retrieval-reviewer` subagent to inspect that run. Keep orchestration in the main
conversation. This skill does not certify model safety or end-to-end answer quality.

## Choose the requested operation

Interpret `$ARGUMENTS` as a mode and named paths, never as an arbitrary shell command.
Default to `quick`. Run `full` only when the user requests the real benchmark.
Use the project interpreter at `${CLAUDE_PROJECT_DIR}/.venv/bin/python`.

From the project root:

```bash
.venv/bin/python scripts/validate_retrieval.py quick
.venv/bin/python scripts/validate_retrieval.py full --baseline data/private/validation/BASELINE_RUN_ID
```

`full` compares one newly executed candidate against an explicitly captured baseline.
It needs the private corpus, source judgement inputs, cached pinned models and DuckDB FTS.
It rebuilds chunks and indexes inside the new run. Missing dependencies or data are
NOT RUN. Do not substitute synthetic inputs or historical CSV files.

If no baseline exists, report that limitation. When the user explicitly requests
baseline creation, run:

```bash
.venv/bin/python scripts/validate_retrieval.py baseline
```

Capture the baseline on the intended reference code before candidate edits. Baseline
creation records scores but does not compare versions. Never silently promote a
candidate, overwrite a baseline or change the policy to make a failing run pass.
Optional `--config path.json` overrides retrieval settings. Baseline and candidate
must have the same corpus, queries, relevance judgements, scoring implementation
and statistical policy.

## Use only this execution's evidence

The command's final JSON supplies `run_id`, `run_dir` and `report`. Retain its exit
code. Exit 0 means the requested scope passed, 1 means failure, and 2 means incomplete.
Read that exact run's manifest, checks, test counts, comparisons and report.
Never select evidence by globbing for the latest directory or by reading old reports.

Before presenting results, run:

```bash
.venv/bin/python scripts/validate_retrieval.py verify --run RUN_DIR --expected-run-id RUN_ID --current
```

Use the exact paths and ID returned by the preceding execution, with shell quoting.
If verification fails, report invalid evidence. If execution never produced a sealed
run, report NOT RUN. Do not infer a pass from an empty test log or missing report.

## Request the second review

Explicitly invoke `retrieval-reviewer` in its own context. Provide the exact run ID,
absolute run directory, policy path and relevant source paths. The manifest contains
the code-change summary. Do not give it a desired verdict or your proposed conclusion.
The reviewer reads files only. Hash checks, test counting and statistical decisions
belong to the validator. If the reviewer is unavailable, report review NOT RUN while
preserving the machine results.

The main agent may save its returned text to `RUN_DIR/review.md`. This is an optional
sidecar excluded from the machine evidence seal. Label it as agent commentary. The
reviewer and main agent must not edit the sealed machine evidence. A requested fix
requires a new validation run. Do not repair code automatically as part of validation.

## Report precisely

State the run ID, requested scope, machine status, actual test counts, real benchmark
status and reviewer completeness. Link the run report and summarize actionable findings.
Keep machine status and agent findings separate, including when they disagree.

For a completed full comparison, show overall and lexical, semantic, mixed and
aggregation results. Delta is candidate minus baseline. Use the machine effect field:

- REGRESSION: paired interval entirely below zero.
- IMPROVEMENT: paired interval entirely above zero.
- INCONCLUSIVE: interval includes zero. No clear difference was detected.

The default policy adjusts five comparisons using Bonferroni, family alpha 0.05,
20,000 bootstrap draws and seed 42. Its individual intervals are 99%. Do not describe
them as unadjusted 95% intervals. Statistical coverage remains approximate.
The single-configuration bootstrap half-width is descriptive, never a fixed 0.08 gate.
PASS does not establish equivalence, safety or the absence of regression.
Quick and hook checks leave the real benchmark comparison NOT RUN.
Raw logs and scores remain private. Share only the allowlisted aggregate report.
