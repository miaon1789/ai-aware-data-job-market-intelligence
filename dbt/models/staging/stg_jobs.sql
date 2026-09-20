-- One row per retained advertisement. Upstream Python owns fuzzy deduplication,
-- contact redaction, geographic normalisation and role/skill classification.
select
    nullif(trim(job_id), '') as job_id,
    nullif(trim(city), '') as city,
    nullif(trim(analysis_role), '') as role,
    nullif(trim(seniority), '') as seniority,
    cast(posted_at as date) as posted_at
from {{ source('pipeline', 'jobs') }}
