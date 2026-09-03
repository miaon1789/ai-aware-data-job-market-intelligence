# Human Annotation Guide

The annotation package contains private job text. Run Doccano locally and do

> **Doccano stores the advertisement text it is shown.** Running it in
> Docker puts that text in the `doccano-db` volume, outside `data/private/`
> and outside anything that walks the working tree. Include it when purging:
> `python scripts/purge_private_data.py --purge-all --include-docker-volumes`.
> Doccano also binds port 8000 by default, which collides with the retrieval
> API; run one or the other, or remap one of them.
not upload these files to a hosted annotation service or GitHub.

Official Doccano JSONL workflow:
<https://doccano.github.io/doccano/tutorial/>

## Prepare tasks

```bash
python scripts/prepare_annotations.py --skill-documents 80
```

The first-pass current snapshot uses every eligible document after the company
cap and an 80-document subset for skill spans. No company contributes more than
three advertisements. Train, development and test splits are company-disjoint
and frozen in the private `manifest.csv`. Doccano task files contain only a
`text` field; exports are matched back to the manifest using a text hash after
annotation.

Do not rename or edit task text after import. The export processor deliberately
rejects text that does not match the frozen manifest.

## Optional AI-assisted suggestions

You can generate first-pass suggestions for train/dev document labels:

```bash
python scripts/suggest_document_labels.py
```

This writes:

```text
data/private/annotations/exports/documents-ai-suggestions.jsonl
data/private/annotations/document_label_suggestions_review.csv
```

These are not human labels. Use them only as review aids in Doccano, edit them
where they are wrong, and do not report them as manual annotation in public
project summaries.

## Optional LLM-assisted audit

LLM output is a semantic review aid, not ground truth. Use it to label the full
private train/dev set automatically, identify disagreements, and select a
stratified human-audit sample. Do not manually label the whole dataset unless
the research question is specifically about annotation.

Preview prompts locally without sending private text to an API:

```bash
python scripts/llm_audit_document_labels.py \
  --document-export data/private/annotations_expanded/exports/documents-annotator-1.business_context_v3_candidate.jsonl \
  --limit 5
```

Run the OpenAI-compatible endpoint only after setting `LLM_API_KEY` and
`LLM_MODEL` in `.env`:

```bash
python scripts/llm_audit_document_labels.py \
  --document-export data/private/annotations_expanded/exports/documents-annotator-1.business_context_v3_candidate.jsonl \
  --run-api \
  --acknowledge-private-text \
  --limit 50
```

The resulting `documents-llm-audit.jsonl` includes Doccano-compatible `text`
and `label` fields, so it can be passed to `process_annotations.py` as a second
reviewer:

```bash
python scripts/process_annotations.py \
  --manifest data/private/annotations_expanded/manifest.csv \
  --document-export data/private/annotations_expanded/exports/documents-annotator-1.business_context_v3_candidate.jsonl \
  --document-export data/private/annotations_expanded/exports/documents-llm-audit.jsonl \
  --output data/private/annotations_expanded/processed_human_llm_audit \
  --report reports/human_llm_audit_annotation_quality.json
```

The prompt asks for two layers:

- observed labels based only on the visible API excerpt;
- role-based coding inference, where the model may use common labour-market
  knowledge.

Keep these layers separate in analysis and public summaries.

After LLM audit, adjudicate only the disagreement rows:

```bash
python scripts/summarize_llm_audit.py
python scripts/prepare_label_adjudication.py
```

Open the private
`data/private/annotations_expanded/label_adjudication_with_source.csv` file and
fill only the empty `final_*` columns. Non-disputed final columns are pre-filled
from the human-reviewed label. When the sheet is complete, apply it:

```bash
python scripts/apply_label_adjudication.py
```

The apply step deliberately fails if any disputed final label is blank. The
completed normalized labels remain private, while
`reports/adjudicated_annotation_summary.json` is text-free and publishable.

## Project 1: role, entry fit, AI signal and coding signal

Create a multi-label text-classification project and import the train and dev
files first:

```text
document_classification_train.jsonl
document_classification_dev.jsonl
```

Keep `document_classification_test.jsonl` unopened until the annotation guide,
skill dictionary and modelling choices are frozen. Define these labels:

You can import the labels directly from:

```text
data/private/annotations/doccano_document_labels.json
```

```text
ROLE::Data Analyst / BI
ROLE::Data Engineer
ROLE::Data Scientist / ML
ROLE::AI / Automation
ROLE::Other / Mixed

ENTRY_FIT::Likely entry-level
ENTRY_FIT::Experienced role
ENTRY_FIT::Unclear

AI_SIGNAL::Explicit GenAI / LLM tool
AI_SIGNAL::Explicit AI / ML system work
AI_SIGNAL::Workflow automation
AI_SIGNAL::AI-adjacent but vague
AI_SIGNAL::No AI signal
AI_SIGNAL::Unclear

CODING_SIGNAL::Standalone coding skill
CODING_SIGNAL::Automation / scripting
CODING_SIGNAL::Production engineering
CODING_SIGNAL::No coding signal
```

Select exactly one `ROLE::`, one `ENTRY_FIT::`, one `AI_SIGNAL::` and one
`CODING_SIGNAL::` label for every task.

Role rules:

- Prefer actual responsibilities over title wording.
- Data Analyst: reporting, BI, dashboards, descriptive analysis and stakeholder
  insights are the central work.
- Data Engineer: pipelines, warehouses, transformations and platform reliability
  are central.
- Data Scientist / ML: statistical experiments, predictive modelling or ML
  analysis are central.
- AI / Automation: GenAI, LLM, agentic workflow, process automation or AI
  solution delivery is central.
- Other / Mixed: genuinely mixed, insufficient, or outside these families.

Entry-fit rules:

- Likely entry-level: explicit graduate, junior, internship, entry-level,
  associate or approximately zero to two years of experience; training or a
  structured graduate program also counts.
- Experienced role: explicit senior, lead, manager, principal, staff, architect,
  substantial ownership, mentoring, people leadership, or three-plus years of
  required experience.
- Unclear: evidence is absent, contradictory, or only implied by vague wording.

AI-signal rules:

- Explicit GenAI / LLM tool: names or directly requires tools/concepts such as
  ChatGPT, Claude, Copilot, Cursor, Gemini, LLM, GenAI, RAG, LangChain, prompt
  engineering or agents.
- Explicit AI / ML system work: directly names machine learning, deep learning,
  computer vision, NLP, model training/evaluation, MLOps, AI platforms or AI
  engineering as role work, even when no GenAI/LLM tool is named.
- Workflow automation: asks for AI-assisted workflow automation, agent
  workflows, n8n, Zapier or similar business/process automation as role work.
  Do not use this label for ordinary report scheduling, SQL jobs, macros or
  generic "process improvement and automation" unless AI or agentic workflow
  tooling is part of the role.
- AI-adjacent but vague: role-specific wording says AI-driven, AI-enabled,
  intelligent automation, AI adoption or similar without a concrete tool, model,
  workflow or responsibility.
- No AI signal: no reliable role-specific AI, GenAI, ML or AI-assisted
  automation signal appears in the excerpt. Employer/product boilerplate does
  not count, for example "Salesforce is the #1 AI CRM", "AI-infused platform",
  "as a leader in AI", "investing in data, digital and AI", or data-centre
  infrastructure described as supporting AI workloads.
- Unclear: wording is too ambiguous to distinguish from ordinary analytics or
  generic company branding.

Coding-signal rules:

- Standalone coding skill: Python, R, SQL or another language is requested as a
  skill by itself, without clear automation or production engineering framing.
- Automation / scripting: code is framed as automating workflows, reports,
  data checks, integration tasks or repetitive operations.
- Production engineering: code is framed around deployable systems, pipelines,
  APIs, CI/CD, testing, monitoring, cloud infrastructure or reliability.
- No coding signal: no concrete coding language, scripting, engineering or SQL
  signal appears in the excerpt. Do not infer coding from prestige, employer,
  title, title seniority or general industry expectations unless the excerpt
  provides textual evidence in responsibilities or requirements. A title such
  as Data Engineer, Solution Architect, Business Analyst, Network Engineer or
  Systems Engineer is not enough by itself.

## Project 2: skill spans and context

Create a sequence-labeling project and import the corresponding train/dev files.
Keep `skill_sequence_labeling_test.jsonl` unopened until extraction rules are
frozen. Define:

You can import the labels directly from:

```text
data/private/annotations/doccano_skill_labels.json
```

```text
SKILL_REQUIRED
SKILL_PREFERRED
SKILL_MENTIONED
SKILL_NEGATED
SKILL_ALTERNATIVE
```

Highlight the shortest complete phrase naming the skill. Include product names
such as `Power BI`, but exclude surrounding phrases such as `strong experience
with`. Label the context expressed in the same sentence:

- REQUIRED: must, essential, required or clearly mandatory.
- PREFERRED: desirable, preferred, highly regarded, bonus or nice-to-have.
- NEGATED: explicitly not required.
- ALTERNATIVE: one option in an `X or Y` requirement.
- MENTIONED: present without a reliable requirement signal.

Do not infer a skill that is not written. Do not label employer names, degrees,
years of experience or generic traits such as `passion`.

## Quality protocol

- Annotate without looking at model predictions.
- Freeze the test split before changing extraction rules.
- Have a second person independently label at least 30 to 50 tasks.
- Resolve disagreements only after both first-pass annotations are complete.
- Report document-label agreement and exact-span F1; do not report only model
  accuracy.
- Never publish the private task files, manifest or raw Doccano exports unless a
  data licence explicitly permits it.

## Validate exports

Export completed annotations from Doccano as JSONL and keep those files under
`data/private/annotations/exports/`. Process train/dev exports before touching
the test tasks:

```bash
python scripts/process_annotations.py \
  --document-export data/private/annotations/exports/documents-annotator-1.jsonl \
  --document-export data/private/annotations/exports/documents-annotator-2.jsonl \
  --skill-export data/private/annotations/exports/skills-annotator-1.jsonl \
  --skill-export data/private/annotations/exports/skills-annotator-2.jsonl
```

Each repeated argument represents another export or annotator. When Doccano
includes a username, it is replaced with a one-way pseudonymous ID. Otherwise,
the export filename identifies the annotator, so use a stable distinct filename
for each person.

The processor fails on unknown labels, missing document-label dimensions,
invalid span offsets, duplicate annotations, non-manifest text and test-split
records.
It writes text-free normalized labels to the ignored
`data/private/annotations/processed/` directory. The aggregate-only
`reports/annotation_quality.json` contains coverage, label distributions,
document Cohen's kappa for role, entry-fit, AI-signal and coding-signal labels,
and pairwise exact-span F1; it contains no job text, job IDs or annotator
names.

Only after the guide and modelling choices are frozen should the test set be
annotated and processed with the explicit `--include-test` flag. That flag is
an audit signal, not a convenience default.
