select
	sample.user_id as user_id,
	sample.create_time as create_time,
	sample.label as log_type,
	kyc.watchlists as watchlists,
	kyc.holdings as holdings,
	kyc.country as country,
	kyc.prefer_bid as prefer_bid,
	sample.push_title as push_title,
	sample.push_content as push_content,
	sample.item_code as item_code,
	sample.tags as item_tags,
	sample.submit_type as submit_type
from
	(
		select
			Behavior.user_id as user_id,
			Behavior.create_time as create_time,
			Behavior.log_type as label,
			Item1.item_id as item_id,
			Item1.push_title as push_title,
			Item1.push_content as push_content,
			Item2.item_code as item_code,
			Item2.tags as tags,
			Item2.submitType as submit_type
		from
			(
				select
					user_id,
					push_id,
					create_time,
					log_type
				from
					hx_dwd.dwd_mkt_push_detail_ds
				where
					p_date = '{now}'
					and log_type in ('PC', 'PR')
					and user_id > 0
			) Behavior
			inner join (
				select
					push_id,
					item_id,
					push_title,
					push_content
				from
					hx_dim.dim_mkt_push_base_info_ds
				where
					p_date = '{now}'
					and come_from in (
						'999677',
						'999678',
						'999679',
						'999680',
						'999681',
						'999682',
						'999684'
					)
			) Item1 on Behavior.push_id = Item1.push_id
			left join (
				select
					item_id,
					logmap ['stocks'] as item_code,
					logmap ['tags'] as tags,
					logmap ['submitType'] as submitType
				from
					hx_dim.dim_mkt_rec_res_pool_item_hd
				where
					p_date = '{now}'
			) Item2 on Item1.item_id = Item2.item_id
		where
			if(
				Behavior.log_type = 'PC',
				1,
				rand(10)
			) > 0.9
	) sample
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
	) kyc on sample.user_id = kyc.person_id
where
	sample.user_id is not null;