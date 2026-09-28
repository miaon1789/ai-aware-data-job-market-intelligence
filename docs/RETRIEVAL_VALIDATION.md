# Retrieval validation workflow

The project provides a Claude Code skill, a post-edit hook and a read-only reviewer.
They use the same Python validator as CI. The statistical and provenance checks work
without an agent runtime.

## Use it

Install the project development dependencies using `make install`. Full benchmark
runs also need `make install-retrieval`, the private source data, the pinned models
in the local Hugging Face cache and the DuckDB FTS extension. Validation does not
download weights or fall back to synthetic data.

From Claude Code opened at the repository root:

```text
/validate-retrieval quick
/validate-retrieval full --baseline data/private/validation/BASELINE_RUN_ID
```

From a terminal:

```bash
.venv/bin/python scripts/validate_retrieval.py quick
```

Before changing the reference retrieval implementation, capture an explicit baseline:

```bash
.venv/bin/python scripts/validate_retrieval.py baseline
```

Keep the returned `run_dir`. After the candidate change, run:

```bash
.venv/bin/python scripts/validate_retrieval.py full --baseline BASELINE_RUN_DIR
```

Baseline capture executes tests and a fresh benchmark, but makes no comparative
effect claim. It is never created automatically. Existing ablation CSVs cannot be
imported as baselines because their execution provenance was not recorded.
To compare a historical code version, capture the baseline with this validator
present on that intended reference checkout. Do not run both sides on candidate code
and describe the result as a version regression test.

`--config settings.json` accepts retrieval setting overrides, for example
`{"mode": "lexical"}`. The manifest records all effective settings and their hash.
Both versions must use the same statistical policy, query set, corpus, skill table
database, annotation inputs and scoring implementation. The evaluation and golden-set
modules and benchmark capture harness are checked conservatively, including changes
to their documentation.
Resolved relevance must also match. Data, scoring or policy
changes require a new explicit baseline. Each run rebuilds chunks, indexes and
embeddings without sharing vector caches, and records the pinned model revisions.

The validator returns a final JSON object containing the run ID and report paths.
Use exactly those values to verify the result, including after agent review:

```bash
.venv/bin/python scripts/validate_retrieval.py verify \
  --run RUN_DIR --expected-run-id RUN_ID --current
```

`verify` verifies evidence, not model performance. It returns zero for intact
evidence even if that run's recorded check status is FAIL. `--current` additionally
checks current code, inputs, policy, configuration and the scanned published file
set against that run. Later report additions also require a fresh privacy check.

## What the modes mean

| Mode | Scope | Real comparison |
|---|---|---|
| quick | Evaluation, golden set, chunking, routing, privacy and validator acceptance tests | NOT RUN |
| quick --profile hook | Evaluation, golden set, privacy guards and synthetic leak detection | NOT RUN |
| baseline | Quick checks and fresh benchmark capture | NOT RUN |
| full | Quick checks, baseline verification, fresh candidate capture and paired comparison | Runs if prerequisites pass |

Each validation command exits 0 for PASS, 1 for FAIL and 2 for NOT RUN. Any failed
check makes the run FAIL. An incomplete required check prevents PASS. Missing private
data is optional for quick checks and mandatory for baseline/full. The two corpus
privacy tests may skip when data is absent, and that limitation is reported separately.
Unexpected skipped tests, no executed tests, missing reports and zero collected tests
cannot silently pass. Test counts come from JUnit cases, and subprocess exit codes
are retained.

## Statistics

The primary metric is nDCG@10, with delta defined as candidate minus baseline.
`eval/validation_policy.json` declares the policy before execution. Overall, lexical,
semantic, mixed and aggregation are one five-comparison family. The default uses
20,000 percentile bootstrap draws, seed 42 and Bonferroni adjustment from family
alpha 0.05. Thus each displayed comparison uses a 99% interval. Bootstrap coverage
is approximate, especially for small groups. Aggregation nDCG measures retrieval,
not the correctness of counted answers.

- Interval entirely below zero: REGRESSION and FAIL.
- Interval entirely above zero: IMPROVEMENT and PASS, unless another check fails.
- Interval containing zero, including an endpoint of zero: INCONCLUSIVE and PASS,
  unless another check fails. This does not establish equivalence or no regression.

Scores retain full precision for decisions. Display rounding does not control the
gate. Single-configuration interval half-widths are descriptive uncertainty only.
No business tolerance or fixed 0.08 threshold is inferred from observed results.

## Evidence and privacy

Every invocation creates `data/private/validation/<run_id>/` with restrictive directory
permissions. It contains the manifest, checks, command logs, JUnit report and machine
report. Benchmark runs add fresh scores, relevance judgements, chunks and indexes.
The manifest records Git commit, dirty-change summary, tracked and non-ignored source
hashes, input and effective configuration hashes, environment versions, commands,
exit codes and output hashes. Source and input hashes are checked again after execution.
The source snapshot covers `src/`, `scripts/`, `tests/`, `eval/`, `config/`, `.claude/`,
`.github/`, `docs/`, and the root README and build configuration files. Git-ignored
local files are excluded unless already tracked. The Git diff summary is explicitly
labelled as tracked files only. The manifest also lists and counts untracked files
individually, rather than counting a new directory as one addition.
Private `code_changes/tracked.patch` records changes within this scope, including
deletions. `code_changes/untracked/` preserves newly added source files for the reviewer.
These artifacts are sealed with the run and are never part of the public export.
A completion marker binds the manifest hash. Interrupted runs without it are invalid.
The privacy evidence records the scanned paths and hashes, checks for changes before
sealing, and binds the newly generated allowlisted aggregate exports as well.
These hashes detect stale or changed evidence. They are not signed attestations against
someone rewriting both the evidence and its hashes.

Only an allowlisted aggregate `summary.json` and `report.md` are exported beneath
`reports/validation/<run_id>/`. CI uploads those two files, never private logs, source
text, resolved document IDs or per-query scores. The privacy scan checks known IDs and
normalized opening text fragments from every local advertisement. It is a bounded
leak check, not a complete detector of all possible quotations. Synthetic fault tests
exercise the detector without private data.
Exports are staged privately and published as one directory only after the evidence
seal has been written and verified. A sealing failure cannot publish a PASS summary.
Both public files identify the Git commit, whether the working tree contained
uncommitted changes, the source fingerprint, the policy hash and the evaluator hash.
Quick runs record the evaluator hash too. It identifies the scoring implementation
but does not establish benchmark performance.

## Claude Code integration

The skill is `.claude/skills/validate-retrieval/SKILL.md`. It runs validation in the
main conversation and explicitly requests the separate `retrieval-reviewer` afterward.
If review is unavailable, review remains NOT RUN and machine evidence is unchanged.
The reviewer has only Read, Grep and Glob. It receives the run identity and evidence
paths without a requested verdict. It looks for coverage gaps and contradictory or
unsupported interpretations. It cannot execute commands or modify machine status.
The main agent may save returned commentary as `review.md`, which is explicitly
outside the machine evidence seal. This is a second-context review, not organizationally
independent model validation.

The project settings register a PostToolUse hook for Edit and Write. Its Python adapter
filters file paths, takes a nonblocking lock and invokes the hook test profile. It does
not invoke a model or run the real benchmark. Timeout, missing environment and concurrent
checks produce NOT RUN feedback. It does not undo edits and does not observe edits made
through Bash or outside Claude Code. The adapter currently targets macOS and Linux.

Project skill, agent and hook settings are versioned. Personal `settings.local.json`
remains ignored. Restart Claude Code if newly created skill or agent directories are
not discovered in an existing session. CI runs the lightweight command without Claude.
Configure the CI job as a required status check in branch protection to enforce a merge
gate. This repository change does not configure remote branch protection.

References: [skills](https://code.claude.com/docs/en/skills),
[hooks](https://code.claude.com/docs/en/hooks),
[subagents](https://code.claude.com/docs/en/sub-agents).
