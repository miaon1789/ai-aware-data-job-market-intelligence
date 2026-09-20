# Local analytics engineering with dbt and DuckDB

This optional dbt layer rebuilds the project's two analytical summaries from
existing Python pipeline outputs. It runs locally without a cloud account.

## Scope and lineage

```text
Python validation, redaction, near-deduplication, role and skill extraction
  -> jobs_clean.csv + skill_mentions.csv
  -> prepare_dbt.py (narrow input tables + original reference SQL views)
  -> stg_jobs + stg_skill_mentions
  -> mart_role_city + mart_role_skill
  -> data tests and reconciliation against the original views
```

This is a migration of the analytical transformation layer, not the whole
cleaning pipeline. Python still owns near-duplicate detection, contact
redaction, geography, labels and NLP. The existing service and dashboards
continue to use their original database. No runtime dependency on dbt is added.

- `stg_jobs`: one row per retained ad, trimming identifiers/dimensions, converting
  blank strings to nulls and parsing the posting date.
- `stg_skill_mentions`: one row per ad, skill, category and context, removing
  repeated sentence-level mentions. It does not discard negated mentions.
- `mart_role_city`: counts by city, role and seniority.
- `mart_role_skill`: distinct ads containing each skill/context, divided by
  **all ads in that city/role**, including ads with no extracted skills.

Column meanings, grains and generic tests are in `models/schema.yml`.
These marts are internal, unsuppressed analytical tables. They are not public
release tables. The existing `publishing.py` suppression and attribution rules
still apply before any real-data publication.

## Install

From the repository root, using Python 3.11 or newer:

```bash
python3 -m venv .venv-dbt
source .venv-dbt/bin/activate
python -m pip install -e '.[dev]' -r dbt/requirements.txt
```

The dbt versions and DuckDB version in `requirements.txt` are the locally tested
combination. A separate environment avoids changing the retrieval environment.

## Reproduce with fictional data

```bash
python scripts/run_pipeline.py \
  --output-dir data/private/dbt/synthetic_input \
  --reports-dir data/private/dbt/synthetic_reports
python scripts/prepare_dbt.py \
  --input-dir data/private/dbt/synthetic_input \
  --database data/private/dbt/synthetic.duckdb
export DBT_DUCKDB_PATH="$PWD/data/private/dbt/synthetic.duckdb"
export DBT_SEND_ANONYMOUS_USAGE_STATS=false
dbt build --project-dir dbt --profiles-dir dbt
dbt docs generate --project-dir dbt --profiles-dir dbt
dbt docs serve --project-dir dbt --profiles-dir dbt
```

`dbt build` builds models in dependency order and runs their tests. Prefer it to
`dbt run` when validating inputs: `run` alone does not enforce data tests.
Generated docs include the dependency graph and column descriptions.

To check an existing local dataset, supply its processed CSV directory to
`prepare_dbt.py` and choose a separate database under `data/private/dbt`.
Do not point dbt at the existing `job_market.duckdb` file. The preparation script
restricts writes to the dedicated dbt directory and copies only five job columns
and four skill columns, not advertisement text or contact information.

## Verification

```bash
RUN_DBT_TESTS=1 python -m pytest tests/test_dbt_analytics.py -q
```

The integration test generates fictional ads without training a classifier,
uses their supplied role labels, and runs the actual dbt CLI. It:

1. Adds a job without skill mentions and a repeated skill mention, checking that
   the denominator remains correct and repeated mentions do not inflate counts.
2. Builds all models and generates dbt documentation.
3. Verifies both marts against the unchanged reference views from
   `analytics.build_database`, using two-way `EXCEPT ALL`. Skill shares are
   rounded to 12 decimal places only for the comparison.
4. Injects a duplicate job ID, blank city, invalid context and orphan skill
   reference into a temporary test database. It checks that the expected dbt
   tests fail and that a subsequent `dbt build` skips downstream marts.

The negative test intentionally uses `dbt run` once to expose all corrupt rows
to column tests. That step is confined to the temporary fixture. A failed build
is not an atomic database rollback: previous successful mart tables can remain,
so consumers must check build status before accepting a new result.

`not_null`, `unique`, `accepted_values` and `relationships` tests check field and
key contracts. Two reconciliation tests and a share-range test check analytical
behaviour. These tests do not certify that predicted roles are semantically
correct or that the sample represents the whole Australian labour market.

The optional CI job runs this synthetic integration check. No licensed input
or generated dbt database, docs, logs or target files are committed. The local
verification summary is recorded in `../reports/dbt_validation.json`.
