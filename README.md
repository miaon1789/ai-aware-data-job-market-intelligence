# AI-Aware Entry-Level Data Job Market Intelligence

An end-to-end data mining project for analysing Australian entry-level data job
advertisements in the AI-assisted work era. The project asks whether current
entry-level data roles still present coding as a standalone gatekeeping skill,
or increasingly bind it to automation, GenAI/LLM tooling, validation,
stakeholder communication and business problem solving.

The MVP demonstrates licensed API ingestion, validation, deduplication,
dictionary-based skill extraction, weak-supervision labels, LLM-assisted
semantic auditing, clustering, association-rule mining, grouped model
evaluation, SQL analytics and an interactive Streamlit dashboard.

> **Data status:** raw job advertisements remain private. `data/public/`
> contains only cell-suppressed aggregates from Adzuna API excerpts, alongside
> explicit attribution and limitations. The semantic analysis uses a private
> 244-ad train/dev sample with human-reviewed labels, Claude-assisted audit and
> human adjudication of all model disagreements. The repository also includes a
> deterministic synthetic generator for reproducible testing.

The broader design proposal is preserved in
[`README_AI_Data_Job_Market_Intelligence.md`](README_AI_Data_Job_Market_Intelligence.md).

## MVP scope

- Sydney and Melbourne
- Entry-level targeted and AI-signal targeted data-role queries
- Data Analyst/BI, Data Engineer, Data Scientist/ML, AI/Automation and mixed
  role families
- Entry-fit labels: likely entry-level, experienced role or unclear
- AI-era signals: explicit GenAI/LLM tools, workflow automation and vague AI
  language
- Coding signals: standalone coding, automation/scripting or production
  engineering
- Normalised technical, AI-tooling, automation and professional skill
  extraction
- Required, preferred, negated, alternative and mentioned contexts
- Exact and near-duplicate removal
- DuckDB analytics tables
- TF-IDF + logistic regression role-classification baseline
- Grouped cross-validation by company to reduce leakage
- TF-IDF + KMeans latent job-family discovery
- Skill association-rule mining with support, confidence and lift
- LLM-assisted semantic label auditing with strict JSON validation
- Streamlit market explorer and privacy-preserving JD comparison

The Adzuna free API description field is truncated at approximately 500
characters in the collected snapshot. Public skill results therefore measure
mentions in API excerpts; they do not measure complete job requirements or
prove that AI has caused a labour-market shift.

## Current results

The expanded private Adzuna snapshot contains 508 raw rows and 421 clean,
deduplicated Sydney/Melbourne advertisements posted between 2026-03-25 and
2026-06-23. Because 94.77% of descriptions are exactly 500 characters, all
claims are framed as excerpt-level evidence.

The final semantic sample contains 244 labelled train/dev advertisements.
Human labels were audited with Claude on all 244 items. The human-vs-LLM audit
found 71 disagreement rows; manual adjudication changed 33 tasks:

- AI-signal changes: 21
- Coding-signal changes: 8
- Entry-fit changes: 6
- Role changes: 3

Final adjudicated label distribution:

- Role: Other/Mixed 132, Data Analyst/BI 35, AI/Automation 31, Data Engineer
  23, Data Scientist/ML 23
- Entry fit: Experienced 124, likely entry-level 47, unclear 73
- AI signal: No AI signal 163, explicit AI/ML system work 34, AI-adjacent
  vague 26, explicit GenAI/LLM 12, workflow automation 6, unclear 3
- Observed coding signal: No coding signal 162, production engineering 64,
  automation/scripting 15, standalone coding skill 3

Headline interpretation: in this excerpt-level sample, explicit AI/ML or GenAI
language is concentrated in AI/Automation and Data Scientist/ML roles, while
entry-level-targeted and analyst roles more often show no role-specific AI
signal. Observed coding language is strongest for Data Engineer roles. `No
coding signal` means no coding evidence appears in the API excerpt, not that
the real job requires no coding.

Core output files:

- [`reports/adjudicated_annotation_summary.json`](reports/adjudicated_annotation_summary.json)
- [`reports/semantic_label_distribution.csv`](reports/semantic_label_distribution.csv)
- [`reports/semantic_role_ai_signal.csv`](reports/semantic_role_ai_signal.csv)
- [`reports/semantic_role_coding_signal.csv`](reports/semantic_role_coding_signal.csv)
- [`reports/human_llm_audit_annotation_quality.json`](reports/human_llm_audit_annotation_quality.json)
- [`reports/mining_summary.json`](reports/mining_summary.json)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_pipeline.py
streamlit run app/streamlit_app.py
```

Run the checks:

```bash
pytest
ruff check .
```

The pipeline generates `data/synthetic/job_ads.csv`, writes cleaned tables to
`data/processed/job_market.duckdb`, stores the baseline model under
`data/processed/role_classifier.joblib`, and writes evaluation metrics to
`reports/model_metrics.json`.

## Use permitted real data

Provide a CSV with the fields documented in [`docs/DATA_CARD.md`](docs/DATA_CARD.md):

```bash
python scripts/run_pipeline.py --input /path/to/permitted_job_ads.csv
```

Custom input may leave `role_label` blank. In that case the demonstration
classifier is trained on the synthetic fixture. To train and evaluate on a
separate, manually labelled dataset, provide it explicitly:

```bash
python scripts/run_pipeline.py \
  --input /path/to/permitted_job_ads.csv \
  --training-input /path/to/labelled_training_ads.csv
```

Raw data are ignored by Git. Do not commit job descriptions unless their
licence explicitly permits redistribution. The pipeline does not scrape job
boards and does not bypass access controls.

## Adzuna personal-research workflow

The repository includes a bounded Adzuna Australia collector that uses your
own API credentials and keeps row-level records private. It does not make a
network request until credentials are configured.

```bash
cp .env.example .env
# Add ADZUNA_APP_ID and ADZUNA_APP_KEY locally.
python scripts/fetch_adzuna.py
python scripts/run_pipeline.py --input data/private/adzuna/job_ads.csv
python scripts/mine_job_market.py
python scripts/build_semantic_analysis.py
python scripts/build_public_release.py \
  --document-labels data/private/annotations_expanded/processed_adjudicated/document_labels.csv
streamlit run app/public_dashboard.py
```

By default the collector uses `entry_level` and `ai_signal` query groups. Add a
fresh general baseline with:

```bash
python scripts/fetch_adzuna.py \
  --query-group entry_level \
  --query-group ai_signal \
  --query-group general_baseline
```

The real-data public exporter enforces aggregate cells of at least ten and
cannot output job IDs, titles, descriptions, companies, URLs or source text.
The existing `app/streamlit_app.py` is a local analyst interface and must not be
deployed with a real row-level database. See
[`docs/ADZUNA_USAGE.md`](docs/ADZUNA_USAGE.md) for terms, attribution, retention
and deletion procedures.

After private analysis, generate company-disjoint Doccano tasks without exposing
job text publicly. These tasks are for sampled human validation and prompt
auditing, not for manually producing the whole dataset:

```bash
python scripts/prepare_annotations.py
```

The private task files and frozen split manifest are documented in
[`docs/ANNOTATION_GUIDE.md`](docs/ANNOTATION_GUIDE.md).
After annotation, `scripts/process_annotations.py` strictly validates Doccano
exports, keeps normalized labels private, and creates a text-free aggregate
agreement report. Test exports remain locked unless `--include-test` is passed
explicitly.

For LLM-assisted semantic review, first preview the prompts without sending
private excerpts to any API:

```bash
python scripts/llm_audit_document_labels.py --limit 5
```

To run an OpenAI-compatible chat-completions endpoint, set `LLM_API_KEY` and
`LLM_MODEL` in `.env`, then explicitly acknowledge that private API excerpts
will be sent to the configured provider:

```bash
python scripts/llm_audit_document_labels.py \
  --run-api \
  --acknowledge-private-text \
  --limit 50
```

The generated private JSONL also follows the document-annotation export shape,
so it can be validated as a second annotator alongside the human-reviewed labels:

```bash
python scripts/process_annotations.py \
  --manifest data/private/annotations_expanded/manifest.csv \
  --document-export data/private/annotations_expanded/exports/documents-annotator-1.business_context_v3_candidate.jsonl \
  --document-export data/private/annotations_expanded/exports/documents-llm-audit.jsonl \
  --output data/private/annotations_expanded/processed_human_llm_audit \
  --report reports/human_llm_audit_annotation_quality.json
```

LLM outputs are review aids. Final claims should use aggregate results and
sampled human validation, not unverified model labels.

Prepare a private adjudication sheet for only the human-vs-LLM disagreements:

```bash
python scripts/summarize_llm_audit.py
python scripts/prepare_label_adjudication.py
```

Fill the `final_*` columns in
`data/private/annotations_expanded/label_adjudication_with_source.csv` for the
disputed dimensions, then apply the completed adjudication:

```bash
python scripts/apply_label_adjudication.py
```

After processed semantic labels exist, build aggregate semantic-analysis tables:

```bash
python scripts/build_semantic_analysis.py \
  --document-labels data/private/annotations_expanded/processed_adjudicated/document_labels.csv
```

For an openly licensed external skill benchmark:

```bash
python scripts/fetch_skillspan.py
python scripts/evaluate_skillspan.py
```

This reports sentence-level dictionary coverage on the pinned MIT-licensed
SkillSpan test set. It is not presented as Australian market evaluation or as
exact span/context performance.

Current public benchmark results are documented in
[`reports/model_evaluation.md`](reports/model_evaluation.md): precision `0.831`,
recall `0.147`, and F1 `0.250`. The low recall is treated as a limitation and a
concrete next modelling task, not hidden behind synthetic metrics.

## Interpretation

The project measures language in the collected advertisements, not the entire
labour market and not actual hiring decisions. It investigates early evidence
of a shift in skill framing; it does not claim that AI has replaced coding
skills or caused observed job-market changes. The JD comparison describes how
a pasted advertisement differs from similar advertisements in the loaded
sample. It does not evaluate a candidate or infer that a person lacks skills.

## Licence

Code in this repository is licensed under the MIT License. Dataset licences are
separate. The included synthetic dataset is generated by this project and may
be used under the same MIT terms. Third-party datasets must retain their own
licence and attribution.
