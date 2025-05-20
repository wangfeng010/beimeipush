-- 从优化后的临时表中获取结果数据
SELECT 
    user_id,
    create_time,
    log_type,
    watchlists,
    holdings,
    country,
    prefer_bid,
    user_propernoun,
    push_title,
    push_content,
    item_code,
    item_tags,
    submit_type
FROM 
    db_test.tmp_final_result_{now}