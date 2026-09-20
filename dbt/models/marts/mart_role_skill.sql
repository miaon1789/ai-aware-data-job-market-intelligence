-- Denominator includes ALL jobs for this city/role, even jobs with no skill mention.
with denominators as (
    select city, role, count(distinct job_id) as total_ads
    from {{ ref('stg_jobs') }}
    group by city, role
)
select
    j.role,
    j.city,
    m.skill,
    m.category,
    m.context,
    count(distinct m.job_id) as ads_with_skill,
    count(distinct m.job_id) * 1.0 / d.total_ads as share_of_ads
from {{ ref('stg_jobs') }} j
join {{ ref('stg_skill_mentions') }} m using (job_id)
join denominators d on j.role = d.role and j.city = d.city
group by j.role, j.city, m.skill, m.category, m.context, d.total_ads
