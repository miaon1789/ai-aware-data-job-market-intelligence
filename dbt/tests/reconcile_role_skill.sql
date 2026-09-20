with actual as (
    select role, city, skill, category, context, ads_with_skill, round(share_of_ads, 12)
    from {{ ref('mart_role_skill') }}
), expected as (
    select role, city, skill, category, context, ads_with_skill, round(share_of_ads, 12)
    from {{ source('pipeline', 'role_skill_summary') }}
)
(select * from actual except all select * from expected)
union all
(select * from expected except all select * from actual)
