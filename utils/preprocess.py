import json
from functools import partial
from typing import Any, List

import pandas as pd
from loguru import logger
from uniprocess.config import Config
from uniprocess.operations import OP_HUB

from utils.dtypes import InferConfig

MISSING_VALUE = (None, "N/A", "-", "", "NULL", "Null", "null")


def int_max(x: int, max_value: int) -> int:
    return max(x, max_value)


def remove_items(x: List[Any], target_values: List[Any]):
    residual = set(x) - set(target_values)
    if not residual:
        return x
    return [i for i in x if i not in target_values]


def json_object_to_list(x: str, key: str, fail_value: str = "null"):
    try:
        x_obj = json.loads(x)
    except Exception as e:
        logger.error(e)
        logger.debug(f"json parse error.got input {x}.")
        return [fail_value]
    x_obj = json.loads(x)
    y = [z.get(key, fail_value) for z in x_obj]
    return y


def has_intersection(x: List[Any], y: List[Any], exclude=MISSING_VALUE) -> int:
    a = set(x) - set(exclude)
    b = set(y) - set(exclude)
    return int(len(set(a) & set(b)) > 0)


SELF_OP_HUB = {
    "int_max": int_max,
    "json_object_to_list": json_object_to_list,
    "has_intersection": has_intersection,
    "remove_items": remove_items,
}
OP_HUB.update(SELF_OP_HUB)


def run_one_op_pd(x, op):
    """
    对输入的DataFrame执行一个操作。

    该函数根据操作对象中的配置，对DataFrame的指定列应用一个函数，并将结果存储在新的列中。

    Args:
        x (pd.DataFrame): 输入的DataFrame。
        op (object): 包含操作配置的对象，包括输入列、输出列、函数名和函数参数。

    Returns:
        pd.DataFrame: 处理后的DataFrame。
    """
    # 获取输入列、输出列、函数名和函数参数
    col_in = op.col_in
    col_out = op.col_out
    func_name = op.func_name
    parameters = op.func_parameters if op.func_parameters else dict()

    # 使用partial函数创建一个部分应用的函数，固定函数参数
    partial_func = partial(OP_HUB[func_name], **parameters)

    # 如果输入列是一个列表，对每一行应用函数，否则对单列应用函数
    if isinstance(col_in, list):
        x[col_out] = x[col_in].apply(lambda row: partial_func(*row), axis=1)
    else:
        x[col_out] = x[col_in].apply(partial_func)

    return x


def data_preprocess(df: pd.DataFrame, config: Config, infer_config: InferConfig):
    """
    数据预处理函数，用于对输入的DataFrame进行一系列操作。

    :param df: 输入的DataFrame，包含需要处理的数据。
    :param config: 配置对象，包含预处理所需的各种配置信息。
    :param infer_config: 推理配置对象，包含与推理相关的配置信息。
    :return: 预处理后的DataFrame。
    """
    # 遍历配置中的每个处理管道
    for pipe in config.process.pipelines:
        # 遍历管道中的每个操作
        for op in pipe.operations:
            # 如果操作的输入列存在于DataFrame中，则执行该操作
            if op.col_in in df.columns:
                df = run_one_op_pd(df, op)
        # 记录当前处理管道的特征名称
        logger.info(f"processing {pipe.feat_name}")

    # 获取变长特征的最大列数
    max_col_num = infer_config.varlen_max_col_num
    # 获取配置中的特征名称与DataFrame列名的交集
    names_set = set(config.feat_names).intersection(df.columns)
    # 获取模型配置中的变长稀疏特征名称
    varlen_sparse_feat_names = config.model.varlen_sparse_feat_names
    logger.info("start process valen sparse feat")

    # 获取DataFrame列名与变长稀疏特征名称的交集
    feat_names = set(df.columns).intersection(varlen_sparse_feat_names)
    # 遍历每个变长稀疏特征名称
    for feat_name in feat_names:
        # 将特征列展开为多列
        x_explode = df[feat_name].apply(pd.Series)
        # 生成输出列名
        out_names = [feat_name + f"_{i}" for i in range(x_explode.columns.stop)][
            :max_col_num
        ]
        logger.info(f"processing {feat_name}")
        # 生成输入列索引
        in_columns = [i for i in range(x_explode.columns.stop)][:max_col_num]
        # 将展开后的列添加到DataFrame中
        df[out_names] = pd.DataFrame(x_explode[in_columns], index=df.index)
        # 从names_set中移除原始特征名称，并添加新的输出列名
        names_set.remove(feat_name)
        names_set = names_set.union(set(out_names))

    return df
