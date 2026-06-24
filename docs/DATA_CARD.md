# Data Card

## Current dataset

The default dataset is deterministic synthetic data created by
`scripts/generate_synthetic_data.py`. Company names, descriptions and records
are fictional. It exists to test and demonstrate the system, not to support
labour-market claims.

The optional Adzuna collector writes row-level records only to the Git-ignored
`data/private/` tree. Public releases are produced separately by
`scripts/build_public_release.py` and contain aggregate tables only. See
`docs/ADZUNA_USAGE.md`.

The current private Adzuna case-study snapshot contains 508 raw rows and 421
clean deduplicated Sydney/Melbourne API excerpts. The semantic analysis uses a
244-ad train/dev sample with human-reviewed labels, Claude-assisted audit and
manual adjudication of all human-vs-LLM disagreement rows. Public releases do
not contain raw descriptions, job IDs, company names or source URLs.

## Input schema

Required CSV columns:

| Field | Type | Meaning |
| --- | --- | --- |
| `job_id` | string | Stable source identifier |
| `title` | string | Advertised job title |
| `description` | string | Job-description text |
| `city` | string | Currently Sydney or Melbourne |
| `company` | string | Employer or anonymised employer ID |
| `posted_at` | date | Advertisement publication date |
| `source` | string | Dataset or API name |
| `source_url` | string | Source URL where permitted |
| `role_label` | string/null | Human label used only for supervised evaluation |

Private Adzuna snapshots also retain `collection_query`, `collection_group` and
`retrieved_at` for local provenance. `collection_group` identifies the sampling
strategy, such as `entry_level`, `ai_signal`, `general_baseline` or `custom`.
Collection queries are discarded before public aggregation; only the API access
date range is retained in public metadata for attribution.

`role_label` must be one of `Data Analyst`, `Data Analyst / BI`, `Data
Scientist`, `Data Scientist / ML`, `Data Engineer`, `ML/AI Engineer`, `AI /
Automation`, or `Other / Mixed` when supplied. The deterministic synthetic
fixture currently covers the original three broad classes; real evaluation must
report the class support present in the manually labelled sample.

## Required provenance for real data

Before loading third-party data, record:

- source owner and retrieval date;
- licence or terms URL and version/date checked;
- collection query, geography and time window;
- permitted uses, redistribution rules and attribution text;
- retention/deletion requirements;
- whether descriptions contain recruiter names, emails or phone numbers.

The preprocessing layer masks common email and Australian phone-number patterns
before processed text is stored. This is a defence-in-depth measure, not a
guarantee that all personal information will be detected. Review source data
before analysis and do not commit raw descriptions unless redistribution is
explicitly permitted. Public accessibility, a Kaggle download page, or an
allowed `robots.txt` path is not by itself a licence.

## Known limitations

- Online advertisements do not represent all hiring.
- Query design and source coverage affect every reported count.
- Job boards may contain duplicated or expired advertisements.
- Skill mentions do not prove that a skill is genuinely required in practice.
- AI-tool mentions do not prove AI adoption inside the employer; they only show
  advertised language in the collected API excerpts.
- Synthetic data produce optimistic, non-market model metrics.
- Adzuna search responses may contain shortened rather than complete job
  descriptions.
