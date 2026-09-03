# Personal Job-Search Workflow

A local, personal-use workflow that collects **full-text Australian data / IT / ML
job ads**, merges multiple sources, filters to relevant roles, and loads them into
the existing analysis pipeline and Streamlit dashboard.

It is built on top of the main project (see [`README.md`](../README.md)) and reuses
its schema, skill extraction, seniority classification and dashboard. All row-level
data stays private under `data/private/` (Git-ignored). Nothing here is intended for
public release.

> Scope: Sydney / Melbourne and the rest of Australia (incl. remote), for
> entry-level and beyond across data, software/IT and ML/AI roles.

---

## Why these data sources

Getting Australian, full-text, legally-accessible job ads is genuinely hard. The
big boards (Seek, LinkedIn, Indeed) prohibit scraping. This workflow therefore
combines two complementary sources, each covering the other's weakness:

| Source | Coverage | Full text? | Entry-level? | Key needed |
| --- | --- | --- | --- | --- |
| **Company ATS** (Greenhouse / Ashby / Lever) | Companies that self-host on these boards | ✅ Yes (median ~5,600 chars) | ⚠️ Skews senior | No (public APIs) |
| **Adzuna** | Broad AU market incl. graduate roles | ❌ Truncated at ~500 chars | ✅ Good breadth | Yes (free) |

> **JSearch** (`jsearch.p.rapidapi.com`, via `scripts/fetch_jsearch.py`) was
> evaluated and **has no Australian coverage** (its Google-for-Jobs backend is empty
> for AU). The code is kept only for non-AU markets. Do not rely on it for Australia.

The result: ATS supplies complete job descriptions (great for understanding real
skill requirements and resume gaps); Adzuna supplies the breadth of actually-
applicable entry-level roles.

---

## One-time setup

```bash
make install            # or: python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env    # then edit .env
```

Add to `.env` (this file is Git-ignored — never commit real keys):

```
ADZUNA_APP_ID=...       # free from https://developer.adzuna.com/
ADZUNA_APP_KEY=...
# JSEARCH_API_KEY=...    # optional; not useful for Australia
```

The ATS collector needs **no** key.

---

## Quick start (full refresh)

```bash
# 1. Full-text roles from public company ATS boards (no key)
python scripts/fetch_ats.py

# 2. Entry-level breadth from Adzuna (add IT/ML queries beyond the built-in groups)
python scripts/fetch_adzuna.py --max-retries 6 \
  --query "graduate software engineer" \
  --query "junior software developer" \
  --query "graduate developer" \
  --query "graduate machine learning engineer" \
  --query "junior machine learning engineer"

# 3. Merge both sources into one input (dedupes by job_id)
python scripts/merge_job_ads.py \
  --input data/private/ats/job_ads.csv \
  --input data/private/adzuna/job_ads.csv \
  --output data/private/merged/job_ads.csv

# 4. Run the pipeline, keeping only data/IT/ML roles
python scripts/run_pipeline.py --input data/private/merged/job_ads.csv --domain-filter

# 5. Explore + compare your resume/JD
streamlit run app/streamlit_app.py     # http://localhost:8501
```

---

## The scripts

### `scripts/fetch_ats.py` — full-text ATS collector
Fetches public company boards, keeps only Australian roles, writes full-text ads.

```bash
python scripts/fetch_ats.py                       # verified seed companies
python scripts/fetch_ats.py --save-raw-responses  # also keep raw JSON privately
python scripts/fetch_ats.py \
  --company greenhouse:databricks \
  --company ashby:Airwallex:"Airwallex" \
  --company lever:<token>
```

- `--company provider:token[:Display Name]` — repeatable. The **token** is the slug
  in the company's careers URL: `boards.greenhouse.io/<token>`,
  `jobs.ashbyhq.com/<token>`, `jobs.lever.co/<token>`.
- Default seed list (verified to expose AU roles) lives in `DEFAULT_COMPANIES` in
  `src/job_market_intelligence/ats.py`: Databricks, MongoDB, Stripe, Culture Amp,
  Datadog, Twilio, Figma, Elastic (Greenhouse) and Airwallex (Ashby).
- Output: `data/private/ats/job_ads.csv` (+ `_provenance.json`).

### `scripts/fetch_adzuna.py` — entry-level breadth
See [`ADZUNA_USAGE.md`](ADZUNA_USAGE.md) for terms/attribution. Uses your key.

- Built-in query groups: `entry_level`, `ai_signal`, `general_baseline`
  (`--query-group`, repeatable). Add ad-hoc terms with `--query` (repeatable).
- `--max-retries 6` is recommended (the API can intermittently drop mid-run).
- Output: `data/private/adzuna/job_ads.csv`.

### `scripts/merge_job_ads.py` — combine sources
```bash
python scripts/merge_job_ads.py --input <a.csv> --input <b.csv> --output <merged.csv>
```
Aligns columns, dedupes by `job_id` (earlier `--input` wins on collisions — put ATS
first so its full text is preferred). Output default: `data/private/merged/job_ads.csv`.

### `scripts/run_pipeline.py --domain-filter` — analyse
Runs age filtering, dedup, seniority classification and skill extraction, then (with
`--domain-filter`) drops non data/IT/ML roles. Writes:

- `data/processed/jobs_clean.csv` — cleaned roles (with `source_url` to apply)
- `data/processed/skill_mentions.csv` — extracted skills per job
- `data/processed/domain_filtered_out.csv` — what the domain filter removed (review it)
- `data/processed/job_market.duckdb` — tables for the dashboard
- `reports/data_quality.json`, `reports/model_metrics.json`

Omit `--domain-filter` to keep every role.

---

## How the pieces work

- **All-Australia city support** — `src/job_market_intelligence/locations.py`
  normalises locations to canonical AU cities + `Remote (Australia)` / `Other
  (Australia)`, and excludes same-named foreign cities (Melbourne FL, Perth
  Scotland) and New Zealand. The pipeline schema accepts these values.
- **Domain relevance filter** — `src/job_market_intelligence/relevance.py` keeps a
  role if its title is clearly technical (software, data scientist, ML engineer, AI,
  Databricks, …) or it mentions ≥2 technical skills; it drops clearly non-technical
  professions (nursing, civil/mechanical engineering, banking analyst programs,
  sales, teaching). Check `domain_filtered_out.csv` and tune the patterns if needed.
- **Extra columns** — the collectors also record `is_remote`, `employment_type`,
  `department`/`api_city`, salary (Adzuna) in the private CSVs for eyeballing; these
  are ignored by the pipeline but handy when browsing.

---

## Using the dashboard

```bash
streamlit run app/streamlit_app.py     # http://localhost:8501
```

- **Market explorer** — browse roles by city / role / skill.
- **JD comparison** — paste a job description (or your resume) to see how it differs
  from similar ads in the loaded sample. Useful for spotting skill gaps and deciding
  which roles to target. It describes language differences; it does not score you.

To apply: open `data/processed/jobs_clean.csv` and follow the `source_url` column.

---

## Refreshing & quotas

- Re-run the Quick-start block whenever you want fresh roles. Adzuna's free tier is
  generous but bounded — the default run is ~72 API calls; subset with
  `--query-group` / `--city` to spend less.
- ATS boards are public and unauthenticated; be polite (the collector rate-limits by
  default). Add companies over time via `--company`.

---

## Privacy & ethics

- Everything under `data/private/` is Git-ignored. Do not commit row-level ads.
- Respect each source's terms (`docs/ETHICS.md`, `docs/ADZUNA_USAGE.md`). This
  workflow uses only official APIs and public boards — no scraping of Seek/LinkedIn.
- For personal job-search analysis only, not candidate ranking or redistribution.
```
