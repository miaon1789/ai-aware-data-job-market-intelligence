-- Grain: city, role, seniority. Counts describe the collected sample.
select city, role, seniority, count(*) as job_count
from {{ ref('stg_jobs') }}
group by city, role, seniority
