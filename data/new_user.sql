select
    t.user_id as user_id,
    kyc.watchlists as watchlists,
    kyc.holdings as holdings,
    kyc.country as country,
    kyc.prefer_bid as prefer_bid
from
    (
        select
            a1.user_id
        from
            (
                select
                    distinct user_id as user_id
                from
                    hx_dwd.dwd_crd_mob_login_info_hs
                where
                    p_date = '{today}'
            ) a1
            left join (
                select
                    user_id
                from
                    hx_dim.dim_crd_mob_user_info_dd
                where
                    p_date = '{last_1_day}'
            ) a2 on a1.user_id = a2.user_id
        where
            a2.user_id is null
    ) t
    left join (
        select
            person_id,
            max(
                case
                    when l_code = 'BM79'
                    and l_value is not null then l_value
                    else null
                end
            ) as active_time,
            max(
                case
                    when l_code = 'BM176'
                    and l_value is not null then l_value
                    else null
                end
            ) as watchlists,
            max(
                case
                    when l_code = 'BM210'
                    and l_value is not null then l_value
                    else null
                end
            ) as holdings,
            max(
                case
                    when l_code = 'BM55'
                    and l_value is not null then l_value
                    else null
                end
            ) as country,
            max(
                case
                    when l_code = 'BM70'
                    and l_value is not null then l_value
                    else null
                end
            ) as prefer_bid
        from
            db_dws.dws_crd_lb_v_dd
        where
            p_date = '{now}'
            and l_code in ('BM79', 'BM176', 'BM210', 'BM55', 'BM70')
            and person_id > 0
        group by
            person_id
    ) kyc on t.user_id = kyc.person_id
where
    t.user_id is not null;