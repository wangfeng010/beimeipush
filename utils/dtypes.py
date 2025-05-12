from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class ModelArgs(BaseModel):
    emb_dim: int = 8
    query_dim_hidden: int = 64
    user_dim_hidden: int = 64
    dim_out: int = 32
    margin: float = 1.0
    bias: bool = True
    activation: str = "relu"


class Tag(BaseModel):
    score: Optional[int] = 0
    tagId: Optional[str] = None
    name: Optional[str] = None
    type: Optional[int] = 0
    parentId: Optional[str] = None


class Stock(BaseModel):
    market: Optional[str] = None
    score: Optional[int] = 0
    code: Optional[str] = None
    name: Optional[str] = None
    type: Optional[int] = 0
    parentId: Optional[str] = None


class ProperNoun(BaseModel):
    score: Optional[int] = 0
    name: Optional[str] = None
    id: Optional[int] = 0
    type: Optional[int] = None
    parentId: Optional[str] = None


class ItemVector(BaseModel):
    code: int
    item_id: str
    success: bool
    item_vector: List[float] = Field(default=list())
    usedTime: int = Field(default=0)


class User(BaseModel):
    docScore: float
    user_id: str
    docId: str = None

    @model_validator(mode="before")
    def default_doc_id(cls, values):
        if "docId" not in values or values["docId"] is None:
            values["docId"] = values.get(
                "user_id",
            )
        return values


class TargetGroup(BaseModel):
    code: int
    usedTime: int
    success: bool
    data: List[User] = Field(default=list())


class DBResult(BaseModel):
    code: int
    data: List[User]
    usedTime: int
    success: bool


class Item(BaseModel):
    item_id: str
    create_time: str
    push_title: Optional[str] = None
    push_content: Optional[str] = None
    item_code: Optional[str] = None
    come_from: Optional[str] = None
    submit_type: Optional[str] = None
    item_tags: Optional[str] = None


class PushItems(BaseModel):
    items: List[Item]


class InferConfig(BaseModel):
    ml_model_dir: str
    ml_model_hdfs_dir: str
    dowload_start_hour: int
    dowload_start_minute: int
    download_max_try: int
    user_data_dir: str
    user_data_sql_dir: str
    new_user_data_sql_dir: str
    new_user_data_dir: str
    user_data_save_days: int
    hour_task_minute: int
    max_user_num_per_iter: int
    user_data_file_num: int
    push_server_url: str
    push_server_url_v3: str
    data_columns: List[str]
    sep: str
    varlen_max_col_num: int
    data_index: str
    header: Optional[int] = None
