from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from light_ctr.data.cross_feature import CROSS_PROCESSOR_DICT
from light_ctr.data.single_feature import SINGLE_PROCESSOR_DICT
from light_ctr.utils.constants import TF_DTYPE_MAPPING


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
    input_sample: Optional[str] = None
    func_list: Optional[List[callable]] = None

    def __post_init__(self):
        ops = list()
        for op in self.operations:
            op_ = OperationConfig(**op)  # type: ignore
            if op_.func_name == "AsType":
                type_name = op_.func_parameters["target_dtype"]
                real_type = TF_DTYPE_MAPPING[type_name]
                op_.func_parameters["target_dtype"] = real_type
            assert (
                op_.func_name in SINGLE_PROCESSOR_DICT.keys()
                or op_.func_name in CROSS_PROCESSOR_DICT.keys()
            ), f"func name {op_.func_name} not found."

            ops.append(op_)
        self.operations = ops


@dataclass
class ProcessConfig:
    pipelines: List[PipelineConfig]

    def __post_init__(self):
        pipelines = list()
        for p in self.pipelines:
            p_ = PipelineConfig(**p)  # type: ignore
            pipelines.append(p_)
        self.pipelines = pipelines


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    with open("/home/jovyan/lightctr/config/feat.yml", "r") as f:
        config = yaml.safe_load(f)
    pprint(config)

    cfg = ProcessConfig(**config)
    pprint(cfg)