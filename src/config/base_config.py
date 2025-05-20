from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class TrainsetConfig(BaseModel):
    data_dir: str
    sep: str
    chunksize: Optional[int] = None
    file_num: int
    header: Optional[int] = None
    label_columns: List[str]
    raw_columns: List[str]
    data_path: Optional[Union[Path, List[Path]]] = None

    def model_post_init(self, __context: Any):
        data_path = Path(self.data_dir)
        if data_path.is_dir():
            self.data_path = sorted(
                list(data_path.glob("*.txt")) + list(data_path.glob("*.csv")),
                reverse=False,
            )  # 日期升序
            if self.file_num:
                self.data_path = self.data_path[-self.file_num :]  # latest files
        elif data_path.is_file() and data_path.suffix.lower() in [
            ".csv",
            ".txt",
            ".tsv",
        ]:
            self.data_path = data_path
        else:
            print(f"[warning] data_path: {data_path}")


class DatasetsConfig(BaseModel):
    trainset: TrainsetConfig


class OperationConfig(BaseModel):
    col_in: Union[str, List[str]]
    col_out: str
    func_name: str
    func_parameters: Dict[str, Any]


class PipelineConfig(BaseModel):
    feat_name: str
    feat_type: str
    vocabulary_size: Optional[int] = None
    embedding_dim: Optional[int] = None
    input_sample: Optional[str] = None
    operations: List[OperationConfig]


class ProcessConfig(BaseModel):
    embedding_dim: Optional[int] = None
    pooling_type: Optional[str] = None
    pipelines: List[PipelineConfig]


class InteractionOperationConfig(BaseModel):
    col_in: List[str]
    col_out: str
    func_name: str
    func_parameters: Dict[str, Any]


class InteractionPipelineConfig(BaseModel):
    feat_name: str
    feat_type: str
    vocabulary_size: Optional[int] = None
    embedding_dim: Optional[int] = None
    operations: List[InteractionOperationConfig]


class InteractionsConfig(BaseModel):
    embedding_dim: Optional[int] = None
    pooling_type: Optional[str] = None
    pipelines: List[InteractionPipelineConfig]


class LabelOperationConfig(BaseModel):
    col_in: str
    col_out: str
    func_name: str
    func_parameters: Dict[str, Any]


class LabelPipelineConfig(BaseModel):
    feat_name: str
    feat_type: str
    vocabulary_size: Optional[int] = None
    embedding_dim: Optional[int] = None
    input_sample: Optional[str] = None
    operations: List[LabelOperationConfig]


class LabelProcessConfig(BaseModel):
    pipelines: List[LabelPipelineConfig]


class SchedulingConfig(BaseModel):
    download_start_hour: int
    download_start_minute: int
    hour_task_minute: int
    download_max_try: int


class DataFormatConfig(BaseModel):
    sep: str
    header: Optional[int] = None
    data_index: str
    data_columns: List[str]
    varlen_max_col_num: int


class DataHandlingConfig(BaseModel):
    user_data_dir: str
    new_user_data_dir: str
    user_data_sql_dir: str
    new_user_data_sql_dir: str
    user_data_save_days: int
    user_data_file_num: int
    max_user_num_per_iter: int
    data_format: DataFormatConfig


class ServicesConfig(BaseModel):
    push_server_url_v2: str
    push_server_url_v3: str


class TrainConfig(BaseModel):
    emb_dim: int = 8
    query_dim_hidden: int = 64
    user_dim_hidden: int = 64
    dim_out: int = 32
    margin: float = 1.0
    bias: bool = True
    activation: str = "relu"


class ModelConfig(BaseModel):
    name: str
    hdfs_relative_path: str  # HDFS相对路径
    local_name: str  # 本地保存的文件名
    type: str = "lightgbm"
    description: str = ""


class ServeConfig(BaseModel):
    """表示环境特定配置文件（如 config_dev.yml）中 infer 部分的配置"""

    scheduling: SchedulingConfig
    data_handling: DataHandlingConfig
    services: ServicesConfig
    models: List[ModelConfig]


class FeaturesConfig(BaseModel):
    process: ProcessConfig
    interactions: InteractionsConfig
    label_process: LabelProcessConfig
    feat_names: Optional[List[str]] = None
    varlen_sparse_feat_names: Optional[List[str]] = None

    def model_post_init(self, __context: Any):
        """生成模型配置"""
        self.feat_names = list()
        categorical_feat_names = list()
        sparse_feat_names = list()
        self.varlen_sparse_feat_names = list()
        dense_feat_names = list()
        embedding_info_dict = dict()
        pipes = self.process.pipelines + self.interactions.pipelines
        for p in pipes:
            self.feat_names.append(p.feat_name)
            if p.feat_type in ("varlen_sparse", "sparse"):
                categorical_feat_names.append(p.feat_name)
                embedding_info_dict[p.feat_name] = (p.vocabulary_size, p.embedding_dim)
                if p.feat_type == "varlen_sparse":
                    self.varlen_sparse_feat_names.append(p.feat_name)
                elif p.feat_type == "sparse":
                    sparse_feat_names.append(p.feat_name)
            else:
                dense_feat_names.append(p.feat_name)


class HDFSConfig(BaseModel):
    api_url: str
    base_model_hdfs_dir: str
    model_version: str


class AppConfig(BaseModel):
    serve: ServeConfig
    datasets: DatasetsConfig
    features: FeaturesConfig
    train: TrainConfig
    hdfs: HDFSConfig
    serve: ServeConfig
