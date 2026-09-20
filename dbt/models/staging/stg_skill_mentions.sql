-- Multiple sentences can mention the same skill. Aggregate analysis needs
-- only one advertisement/skill/category/context combination.
select distinct
    nullif(trim(job_id), '') as job_id,
    nullif(trim(skill), '') as skill,
    nullif(trim(category), '') as category,
    nullif(trim(context), '') as context
from {{ source('pipeline', 'skill_mentions') }}
