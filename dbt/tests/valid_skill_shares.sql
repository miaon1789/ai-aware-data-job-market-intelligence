select * from {{ ref('mart_role_skill') }}
where ads_with_skill <= 0 or share_of_ads <= 0 or share_of_ads > 1
