import json
from typing import List

import requests

BATCH_SVAE_URL = "http://ainvest-recsys-resource-pool-server-14660:12333\
/v1/resources/bulk_add_or_merge_update_or_delete_doc"

QUERY_URL = "http://ainvest-recsys-resource-pool-server-14660:12333\
/v1/resources/script_score_query"


def db_batch_save(
    vector_list: List[List[float]], user_list: List[str]
) -> requests.Response:
    """向量数据批量保存

    Parameters
    ----------
    vector_list : List[List[float]]
        由多个向量组成，单个向量为数值列表
    user_list : List[str]
        用户ID列表

    Returns
    -------
    requests.Response
        _description_
    """
    assert len(vector_list) == len(
        user_list
    ), f"vector number and user number must be equal.\
        but got len(vector)={len(vector_list)} and len(user)={len(user_list)}."
    post_map = dict()
    for uid, vec in zip(user_list, vector_list):
        post_map[str(uid)] = {"user_id": str(uid), "vector": vec}

    content = {"resourceName": "UserVector2", "upsertMap": post_map}
    application_json = {"Content-Type": "application/json"}
    return requests.post(
        BATCH_SVAE_URL, data=json.dumps(content), headers=application_json
    )


def db_query(
    query_vector: List[float], recall_user_num: int, minScore: float = 0.8
) -> requests.Response:
    """基于向量检索接口来圈选TOP用户

    Parameters
    ----------
    query_vector : List[float]
        _description_
    recall_user_num : int
        圈选用户数量
    minScore : float, optional
        _description_, by default 0.8

    Returns
    -------
    requests.Response
        _description_
    """
    content = {
        "resourceName": "UserVector2",
        "size": recall_user_num,
        "minScore": minScore,
        "includeFields": ["user_id"],
        "scriptCode": "cosineSimilarity(params.vector,'vector')",
        "scriptParamMap": {"vector": query_vector},
    }
    application_json = {"Content-Type": "application/json"}

    return requests.post(QUERY_URL, data=json.dumps(content), headers=application_json)
