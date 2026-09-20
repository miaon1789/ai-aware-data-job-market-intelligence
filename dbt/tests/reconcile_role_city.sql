-- The reference view is built by the existing analytics.build_database function.
-- EXCEPT ALL catches missing/extra rows and duplicate output rows in both directions.
with actual as (
    select city, role, seniority, job_count from {{ ref('mart_role_city') }}
), expected as (
    select city, role, seniority, job_count
    from {{ source('pipeline', 'role_city_summary') }}
)
(select * from actual except all select * from expected)
union all
(select * from expected except all select * from actual)
