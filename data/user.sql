select
    t.user_id as user_id,
    kyc.watchlists as watchlists,
    kyc.holdings as holdings,
    kyc.country as country,
    kyc.prefer_bid as prefer_bid,
    kyc.user_propernoun as user_propernoun
from
    (
        select
            user_id
        from
            hx_dim.dim_mkt_push_notify_open_close_hd
        where
            p_date = '{now}'
            and is_open = '是'
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
            ) as prefer_bid,
            max(
                case
                    when l_code = 'BM230'
                    and l_value is not null then l_value
                    else null
                end
            ) as user_propernoun
        from
            db_dws.dws_crd_lb_v_dd
        where
            p_date = '{now}'
            and l_code in ('BM79', 'BM176', 'BM210', 'BM55', 'BM70', 'BM230')
            and person_id > 0
        group by
            person_id
    ) kyc on t.user_id = kyc.person_id
where
    t.user_id is not null;