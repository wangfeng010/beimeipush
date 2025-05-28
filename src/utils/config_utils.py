#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置工具函数，用于加载和处理各种配置
"""

import yaml
from typing import Dict, List, Any, Tuple


def load_data_config(config_path: str = "config/data.yml") -> Dict:
    """加载数据配置"""
    with open(config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    return data_config


def extract_config_info(data_config: Dict, TF_DTYPE_MAPPING: Dict) -> Tuple[str, List[str], List[Any], List[str]]:
    """从配置中提取数据信息"""
    file_pattern = data_config["train_dir"] + "/*.csv"
    
    # 从配置文件中提取列名和默认值类型
    column_names = [list(c.keys())[0] for c in data_config["raw_data_columns"]]
    column_defaults = [TF_DTYPE_MAPPING[list(c.values())[0]] for c in data_config["raw_data_columns"]]
    
    # 从配置文件中提取标签列的名称
    label_columns = data_config['label_columns']
    
    return file_pattern, column_names, column_defaults, label_columns


def load_feature_config(config_path: str = "config/feat.yml", exclude_features: List[str] = None) -> List[Dict]:
    """
    加载特征配置
    
    参数:
    - config_path: 配置文件路径
    - exclude_features: 要排除的特征列表，例如['user_id']
    
    返回:
    - 处理后的特征管道配置列表
    """
    # 如果没有提供要排除的特征，设置默认值
    if exclude_features is None:
        exclude_features = ['user_id']  # 默认排除user_id特征
    
    with open(config_path, 'r') as f:
        feat_config = yaml.safe_load(f)
    
    # 深度复制管道配置，以便安全修改
    pipelines_copy = []
    excluded_pipelines = 0
    
    for pipeline in feat_config['pipelines']:
        # 检查这个管道是否处理的是被排除的特征
        should_exclude = False
        
        # 检查管道的第一个操作是否针对被排除的特征
        if 'operations' in pipeline and pipeline['operations']:
            first_op = pipeline['operations'][0]
            if 'col_in' in first_op and first_op['col_in'] in exclude_features:
                should_exclude = True
                excluded_pipelines += 1
                print(f"排除特征: {first_op['col_in']}")
                continue  # 跳过此管道
        
        # 如果不需要排除，则处理这个管道
        if not should_exclude:
            pipeline_copy = {}
            for key, value in pipeline.items():
                if key == 'operations':
                    operations_copy = []
                    for operation in value:
                        operation_copy = operation.copy()
                        # 如果是FillNaString操作，替换为CustomFillNaString
                        if operation_copy.get('func_name') == 'FillNaString':
                            operation_copy['func_name'] = 'CustomFillNaString'
                        operations_copy.append(operation_copy)
                    pipeline_copy[key] = operations_copy
                else:
                    pipeline_copy[key] = value
            pipelines_copy.append(pipeline_copy)
    
    print(f"特征过滤结果: 总共{len(feat_config['pipelines'])}个管道，排除了{excluded_pipelines}个管道，保留{len(pipelines_copy)}个管道")
    
    return pipelines_copy


def load_train_config(config_path: str = "config/train.yml") -> Dict:
    """加载训练配置"""
    with open(config_path, 'r') as f:
        train_config = yaml.safe_load(f)
    return train_config 