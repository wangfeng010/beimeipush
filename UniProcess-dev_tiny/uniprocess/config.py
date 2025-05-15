from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class OperationConfig:
    """算子的配置参数"""

    col_in: Union[str, List[str]]
    col_out: str
    func_name: str  # 对于operation.py中的函数名
    func_parameters: Optional[Dict] = None


@dataclass
class PipelineConfig:
    """数据管道的配置参数"""

    feat_name: str  # 输入模型的特征名
    feat_type: str  # 数据类型：varlen_sparse, sparse, dense
    operations: List[OperationConfig]
    vocabulary_size: Optional[int] = None
    embedding_dim: Optional[int] = None
    input_sample: Optional[str] = None

    def __post_init__(self):
        ops = list()
        for op in self.operations:
            op_ = OperationConfig(**op)  # type: ignore
            ops.append(op_)
        self.operations = ops


@dataclass
class ProcessConfig:
    pipelines: List[PipelineConfig]
    embedding_dim: Optional[int] = None
    pooling_type: Optional[str] = "sum"

    def __post_init__(self):
        pipelines = list()
        for p in self.pipelines:
            p_ = PipelineConfig(**p)  # type: ignore
            pipelines.append(p_)
        self.pipelines = pipelines


@dataclass
class DatasetConfig:
    data_dir: str
    sep: str
    raw_columns: List[str]
    chunksize: int
    header: bool = True
    file_num: Optional[int] = None
    label_columns: Optional[List[str]] = None

    def __post_init__(self):
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


@dataclass
class DataSetsConfig:
    trainset: DatasetConfig
    valset: Optional[DatasetConfig] = None

    def __post_init__(self):
        if isinstance(self.trainset, dict):
            self.trainset = DatasetConfig(**self.trainset)
        if isinstance(self.valset, dict):
            self.valset = DatasetConfig(**self.valset)


@dataclass
class ModelConfig:
    categorical_feat_names: List[str]  # 所以类型特征：包括varlen_sparse和sparse
    dense_feat_names: List[str]  # 数值特征
    sparse_feat_names: List[str]  # 稀疏特征（枚举型）
    varlen_sparse_feat_names: List[str]  # 变长稀疏特征（列表内包含可枚举元素）
    embedding_info_dict: Dict[str, Tuple[int, int]]


@dataclass
class Config:
    datasets: DataSetsConfig
    process: ProcessConfig  # 普通预处理的配置文件
    interactions: ProcessConfig  # 特征交叉的配置文件
    label_process: ProcessConfig

    def _to_dataclass(self):
        """将字典转化为dataclass"""
        if isinstance(self.datasets, dict):
            self.datasets = DataSetsConfig(**self.datasets)
        if isinstance(self.process, dict):
            self.process = ProcessConfig(**self.process)
        if isinstance(self.interactions, dict):
            self.interactions = ProcessConfig(**self.interactions)
        if isinstance(self.label_process, dict):
            self.label_process = ProcessConfig(**self.label_process)

    def _generate_model_config(self):
        """生成模型配置"""
        self.feat_names = list()
        categorical_feat_names = list()
        sparse_feat_names = list()
        varlen_sparse_feat_names = list()
        dense_feat_names = list()
        embedding_info_dict = dict()
        pipes = self.process.pipelines + self.interactions.pipelines
        for p in pipes:
            self.feat_names.append(p.feat_name)
            if p.feat_type in ("varlen_sparse", "sparse"):
                categorical_feat_names.append(p.feat_name)
                embedding_info_dict[p.feat_name] = (p.vocabulary_size, p.embedding_dim)
                if p.feat_type == "varlen_sparse":
                    varlen_sparse_feat_names.append(p.feat_name)
                elif p.feat_type == "sparse":
                    sparse_feat_names.append(p.feat_name)
            else:
                dense_feat_names.append(p.feat_name)

        self.model = ModelConfig(
            categorical_feat_names=categorical_feat_names,
            sparse_feat_names=sparse_feat_names,
            varlen_sparse_feat_names=varlen_sparse_feat_names,
            dense_feat_names=dense_feat_names,
            embedding_info_dict=embedding_info_dict,
        )

    def __post_init__(self):
        self._to_dataclass()
        self._generate_model_config()
