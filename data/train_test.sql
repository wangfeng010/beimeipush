-- 1. 创建用户行为中间表
CREATE TABLE IF NOT EXISTS db_test.tmp_user_behavior AS
SELECT 
    user_id,
    push_id,
    create_time,
    log_type
FROM 
    hx_dwd.dwd_mkt_push_detail_ds
WHERE 
    p_date = '{now}'
    AND log_type IN ('PC', 'PR')
    AND user_id > 0
    AND create_time >= date_sub('{now}', 7)
    AND create_time < '{now}';

-- 2. 创建推送内容中间表
CREATE TABLE IF NOT EXISTS db_test.tmp_push_content AS
SELECT 
    push_id,
    item_id,
    push_title,
    push_content
FROM 
    hx_dim.dim_mkt_push_base_info_ds
WHERE 
    p_date = '{now}'
    AND come_from IN ('999677', '999678', '999679', '999680', '999681', '999682', '999684');

-- 3. 创建物品信息中间表
create TABLE db_test.tmp_rec_push_item_20250507 as
select
	item_id,
	logmap ['stocks'] as item_code,
	logmap ['tags'] as tags,
	logmap ['submitType'] as submitType
from
	hx_dim.dim_mkt_rec_res_pool_item_hd
where
	p_date = '20250507'
	and item_2nd_type in ('article', 'news_flash', 'ainvest_video')
	and FROM_UNIXTIME(
                CAST(logmap ['createTime'] / 1000 AS BIGINT),
                'yyyyMMdd'
            ) >= date_fmat(
                DATE_SUB(date_fmat('20250507', 'yyyy-MM-dd'), 1),
                'yyyyMMdd'
            )

-- 4. 创建用户特征中间表
CREATE TABLE IF NOT EXISTS db_test.tmp_user_features AS
SELECT 
    person_id,
    max(CASE WHEN l_code = 'BM176' AND l_value IS NOT NULL THEN l_value ELSE NULL END) AS watchlists,
    max(CASE WHEN l_code = 'BM210' AND l_value IS NOT NULL THEN l_value ELSE NULL END) AS holdings,
    max(CASE WHEN l_code = 'BM55' AND l_value IS NOT NULL THEN l_value ELSE NULL END) AS country,
    max(CASE WHEN l_code = 'BM70' AND l_value IS NOT NULL THEN l_value ELSE NULL END) AS prefer_bid,
    max(CASE WHEN l_code = 'BM230' AND l_value IS NOT NULL THEN l_value ELSE NULL END) AS user_propernoun
FROM 
    db_dws.dws_crd_lb_v_dd
WHERE 
    p_date = '{now}'
    AND l_code IN ('BM79', 'BM176', 'BM210', 'BM55', 'BM70', 'BM230')
    AND person_id > 0
GROUP BY 
    person_id;

-- 5. 创建最终结果表（采样在这一步完成）
CREATE TABLE IF NOT EXISTS db_test.tmp_final_result AS
SELECT
    b.user_id,
    b.create_time,
    b.log_type AS log_type,
    u.watchlists,
    u.holdings,
    u.country,
    u.prefer_bid,
    u.user_propernoun,
    p.push_title,
    p.push_content,
    i.item_code,
    i.tags AS item_tags,
    i.submitType AS submit_type
FROM 
    db_test.tmp_user_behavior b
INNER JOIN 
    db_test.tmp_push_content p ON b.push_id = p.push_id
LEFT JOIN 
    db_test.tmp_rec_push_item_20250507 i ON p.item_id = i.item_id
LEFT JOIN 
    db_test.tmp_user_features u ON b.user_id = u.person_id
WHERE 
    b.user_id IS NOT NULL
    AND IF(b.log_type = 'PC', 1, rand(10)) > 0.9;

-- 6. 查看结果
SELECT * FROM db_test.tmp_final_result LIMIT 10;

-- 7. 清理中间表（注释掉，可选执行）
-- DROP TABLE IF EXISTS db_test.tmp_user_behavior;
-- DROP TABLE IF EXISTS db_test.tmp_push_content;
-- DROP TABLE IF EXISTS db_test.tmp_rec_push_item_20250507;
-- DROP TABLE IF EXISTS db_test.tmp_user_features;
-- DROP TABLE IF EXISTS db_test.tmp_final_result;
