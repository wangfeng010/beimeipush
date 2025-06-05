from typing import Literal, Union

MISSING_VALUE = [None, "N/A", "-", "", "NULL", "Null", "null"]
DEFAULT_GROUP_NAME: str = (
    "default_group"  # 假设DEFAULT_GROUP_NAME是一个字符串类型，默认群组名称
)
EmbeddingDim = Union[int, Literal["auto"]]  # 定义嵌入维度的类型，可以是整数或"auto"
DType = Literal["int32", "int64"]  # 定义数据类型的枚举

FEAT_TYPE = Literal["sparse", "dense", "varlen_sparse"]  # 特征类型
