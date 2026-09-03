# Adzuna Personal-Research Workflow

This workflow implements a conservative publication boundary without assuming
permission to redistribute job advertisements. It is an engineering control,
not legal advice.

Terms reviewed: 22 June 2026

- API terms: <https://developer.adzuna.com/docs/terms_of_service>
- Required research attribution target: <https://www.adzuna.com.au/>

The terms expressly list personal research as a permissible use and require
Adzuna attribution wherever salary or vacancy data are published. Terms can
change; re-check them before each new collection period.

## 1. Configure private credentials

```bash
cp .env.example .env
```

Edit `.env` locally. It is Git-ignored.

```text
ADZUNA_APP_ID=...
ADZUNA_APP_KEY=...
```

## 2. Collect a bounded snapshot

```bash
python scripts/fetch_adzuna.py
```

Defaults:

- `entry_level` and `ai_signal` query groups across Sydney and Melbourne;
- one page per query, for 62 API calls with the default groups;
- no more than 50 results per page;
- at least 2.4 seconds between calls;
- at most 100 calls in one run;
- no raw API response retention.

Use a fresh general baseline when the research question needs contrast with
non-targeted data-role advertisements:

```bash
python scripts/fetch_adzuna.py \
  --query-group entry_level \
  --query-group ai_signal \
  --query-group general_baseline
```

Named groups currently mean:

- `entry_level`: graduate, junior, entry-level, internship and related data
  role searches;
- `ai_signal`: AI, GenAI, LLM, automation and related data-role searches;
- `general_baseline`: broad data analyst, data scientist, data engineer,
  machine-learning engineer and analytics engineer searches.

The canonical private CSV and provenance record are written under
`data/private/adzuna/`. Use `--save-raw-responses` only when an audit need
justifies retaining full responses; they remain private and Git-ignored.

The free search response may contain shortened descriptions. Review description
length and extraction quality before claiming full-JD analysis. Do not follow
redirect URLs to scrape another job board. Treat the AI-era research question
as an excerpt-level current-snapshot study unless you have separately licensed
historical job text.

## 3. Run private analysis

```bash
python scripts/run_pipeline.py \
  --input data/private/adzuna/job_ads.csv \
  --max-age-days 90
```

This uses the synthetic labelled fixture for the demonstration classifier. A
separate manually labelled, privately stored dataset can be supplied with
`--training-input`.

The local analyst dashboard loads row-level processed data and must not be
deployed publicly when it contains real advertisements:

The pipeline writes an aggregate-only `reports/data_quality.json`. If it flags
description truncation, all public claims and chart labels must say `API
excerpts`; required/preferred context results must not be presented as complete
job-description analysis.

If no private document labels are supplied, role aggregates come from an
unvalidated synthetic-trained baseline and must be labelled `provisional`.
For the public case study, build the public release from the adjudicated
semantic sample instead:

```bash
python scripts/build_public_release.py \
  --document-labels data/private/annotations_expanded/processed_adjudicated/document_labels.csv
```

```bash
streamlit run app/streamlit_app.py
```

## 4. Build a public aggregate release

```bash
python scripts/build_public_release.py
```

For real data the exporter enforces a minimum cell size of ten. It produces
only:

- role/city/month counts;
- role/city/seniority counts;
- role/city/skill/context counts and rounded shares;
- aggregate provenance and attribution metadata.

It cannot export job IDs, titles, descriptions, companies, URLs, source
sentences or collection queries. Do not manually join those fields back into a
public table.

Review `data/public/` before publishing, then run the aggregate-only dashboard:

```bash
streamlit run app/public_dashboard.py
```

## 5. Retention and termination

Preview private files older than 90 days:

```bash
python scripts/purge_private_data.py
```

Delete the reviewed files:

```bash
python scripts/purge_private_data.py --confirm
```

If API access is terminated, review a complete purge first and then confirm it:

```bash
python scripts/purge_private_data.py --purge-all
python scripts/purge_private_data.py --purge-all --confirm
```

A filesystem purge is not a complete purge. Annotating in Doccano copies the
advertisement text into the container's own database, which lives in the
`doccano-db` Docker volume and survives deleting this repository entirely. The
dry run reports any such volume it finds; include them explicitly:

```bash
python scripts/purge_private_data.py --purge-all --include-docker-volumes
python scripts/purge_private_data.py --purge-all --include-docker-volumes --confirm
```

This removes the `doccano` container and its volume. Export any annotations you
still need first — the volume is the only copy of the Doccano projects.

Also remove aggregate releases and deployed dashboards if required by the
provider's then-current terms.

## Attribution

Use this beside every real-data table, chart, report and dashboard:

```text
Vacancy data sourced from The Adzuna API
(https://www.adzuna.com.au/), accessed [DATE].

Only aggregated results are published. Raw job advertisements are not
redistributed. Results describe the collected sample, not the entire
Australian labour market.
```
