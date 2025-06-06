#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置工具函数，用于加载和处理各种配置
"""

import os
import yaml
from typing import Dict, List, Any, Tuple, Optional, Union


def load_data_config(config_path: str = "config/data.yml") -> Dict[str, Any]:
    """
    加载数据配置
    
    Args:
        config_path: 配置文件路径，默认为config/data.yml
        
    Returns:
        数据配置字典
    
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML解析错误
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    return data_config


def extract_config_info(
    data_config: Dict[str, Any], 
    TF_DTYPE_MAPPING: Dict[str, Any]
) -> Tuple[str, List[str], List[Any], List[str]]:
    """
    从配置中提取数据信息
    
    Args:
        data_config: 数据配置字典
        TF_DTYPE_MAPPING: 数据类型映射字典
        
    Returns:
        包含以下元素的元组:
        - file_pattern: 文件模式
        - column_names: 列名列表
        - column_defaults: 列默认值列表
        - label_columns: 标签列名列表
    
    Raises:
        KeyError: 配置中缺少必要的键
        IndexError: 配置结构不符合预期
    """
    file_pattern = data_config["train_dir"] + "/*.csv"
    
    # 从配置文件中提取列名和默认值类型
    column_names = [list(c.keys())[0] for c in data_config["raw_data_columns"]]
    column_defaults = [TF_DTYPE_MAPPING[list(c.values())[0]] for c in data_config["raw_data_columns"]]
    
    # 从配置文件中提取标签列的名称
    label_columns = data_config.get('label_columns', ['label'])
    
    return file_pattern, column_names, column_defaults, label_columns


def load_feature_config(
    config_path: str = "config/feat.yml", 
    exclude_features: Optional[List[str]] = None,
    exclude_config_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    加载特征配置
    
    Args:
        config_path: 配置文件路径，默认为config/feat.yml
        exclude_features: 要排除的特征列表，如果提供则优先使用此参数
        exclude_config_key: 排除配置的键名，从配置文件中读取排除特征列表
        
    Returns:
        处理后的特征管道配置列表
    
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML解析错误
        KeyError: 配置中缺少必要的键
    """
    # 加载配置文件
    feat_config = _load_yaml_config(config_path)
    
    # 确定要排除的特征列表
    final_exclude_features = _determine_exclude_features(
        feat_config, exclude_features, exclude_config_key
    )
    
    # 处理特征管道配置
    return _process_feature_pipelines(feat_config, final_exclude_features)


def _determine_exclude_features(
    feat_config: Dict[str, Any],
    exclude_features: Optional[List[str]] = None,
    exclude_config_key: Optional[str] = None
) -> List[str]:
    """
    确定要排除的特征列表
    
    Args:
        feat_config: 特征配置字典
        exclude_features: 直接指定的排除特征列表
        exclude_config_key: 配置文件中的排除配置键名
        
    Returns:
        最终的排除特征列表
    """
    # 1. 如果直接提供了exclude_features，优先使用
    if exclude_features is not None:
        print(f"使用直接指定的排除特征: {exclude_features}")
        return exclude_features
    
    # 2. 如果指定了exclude_config_key，从配置文件读取
    if exclude_config_key is not None:
        exclude_features_config = feat_config.get('exclude_features', {})
        if exclude_config_key in exclude_features_config:
            features_to_exclude = exclude_features_config[exclude_config_key]
            print(f"从配置文件读取排除特征 [{exclude_config_key}]: {features_to_exclude}")
            return features_to_exclude
        else:
            print(f"警告: 配置键 '{exclude_config_key}' 不存在，使用默认配置")
    
    # 3. 尝试使用配置文件中的current配置
    exclude_features_config = feat_config.get('exclude_features', {})
    if 'current' in exclude_features_config:
        current_config_key = exclude_features_config['current']
        if current_config_key in exclude_features_config:
            features_to_exclude = exclude_features_config[current_config_key]
            print(f"使用当前配置 [{current_config_key}]: {features_to_exclude}")
            return features_to_exclude
        else:
            print(f"警告: 当前配置键 '{current_config_key}' 不存在")
    
    # 4. 使用默认配置
    if 'default' in exclude_features_config:
        default_features = exclude_features_config['default']
        print(f"使用默认配置: {default_features}")
        return default_features
    
    # 5. 最后的fallback，排除user_id
    print("使用最后的默认值: ['user_id']")
    return ['user_id']


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML解析错误
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"解析YAML配置文件失败: {e}")


def _process_feature_pipelines(
    feat_config: Dict[str, Any], 
    exclude_features: List[str]
) -> List[Dict[str, Any]]:
    """
    处理特征管道配置
    
    Args:
        feat_config: 特征配置字典
        exclude_features: 要排除的特征列表
        
    Returns:
        处理后的管道配置列表
    
    Raises:
        KeyError: 配置中缺少必要的键
    """
    if 'pipelines' not in feat_config:
        raise KeyError("特征配置缺少'pipelines'键")
    
    pipelines = feat_config['pipelines']
    processed_pipelines = []
    excluded_count = 0
    
    # 处理每个管道配置
    for pipeline in pipelines:
        if _should_exclude_pipeline(pipeline, exclude_features):
            excluded_count += 1
            continue
            
        # 处理并添加管道配置
        processed_pipeline = _process_single_pipeline(pipeline)
        processed_pipelines.append(processed_pipeline)
    
    # 打印处理结果
    print(f"特征过滤结果: 总共{len(pipelines)}个管道，"
          f"排除了{excluded_count}个管道，"
          f"保留{len(processed_pipelines)}个管道")
    
    return processed_pipelines


def _should_exclude_pipeline(
    pipeline: Dict[str, Any], 
    exclude_features: List[str]
) -> bool:
    """
    检查管道是否应该被排除
    
    Args:
        pipeline: 管道配置
        exclude_features: 要排除的特征列表
        
    Returns:
        如果管道应该被排除则返回True，否则返回False
    """
    # 1. 检查特征名称是否应该被排除（原有逻辑）
    feat_name = pipeline.get('feat_name', '')
    for exclude_feature in exclude_features:
        if exclude_feature in feat_name:
            print(f"排除特征管道: {feat_name} (包含排除关键词: {exclude_feature})")
            return True
    
    # 2. 检查管道的第一个操作是否针对被排除的特征（原有逻辑）
    if 'operations' in pipeline and pipeline['operations']:
        first_op = pipeline['operations'][0]
        if 'col_in' in first_op and first_op['col_in'] in exclude_features:
            print(f"排除特征管道: {feat_name} (基于输入列: {first_op['col_in']})")
            return True
    
    # 3. 🔧 新增：检查管道中任何操作是否以被排除的特征作为输入
    if 'operations' in pipeline and pipeline['operations']:
        for i, operation in enumerate(pipeline['operations']):
            if 'col_in' in operation and operation['col_in'] in exclude_features:
                print(f"排除特征管道: {feat_name} (操作{i+1}使用了被排除的输入列: {operation['col_in']})")
                return True
    
    # 4. 🔧 新增：检查feat_name是否基于被排除的特征命名
    # 例如：user_propernoun -> user_propernoun_hash, user_propernoun_emb等
    for exclude_feature in exclude_features:
        if feat_name.startswith(exclude_feature + '_'):
            print(f"排除特征管道: {feat_name} (基于被排除特征的衍生特征: {exclude_feature})")
            return True
        # 也检查以exclude_feature结尾的情况
        if feat_name.endswith('_' + exclude_feature):
            print(f"排除特征管道: {feat_name} (基于被排除特征的衍生特征: {exclude_feature})")
            return True
    
    return False


def _process_single_pipeline(pipeline: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个管道配置
    
    Args:
        pipeline: 单个管道配置
        
    Returns:
        处理后的管道配置
    """
    pipeline_copy = {}
    
    for key, value in pipeline.items():
        if key == 'operations':
            pipeline_copy[key] = _process_operations(value)
        else:
            pipeline_copy[key] = value
    
    return pipeline_copy


def _process_operations(operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    处理操作配置列表
    
    Args:
        operations: 操作配置列表
        
    Returns:
        处理后的操作配置列表
    """
    operations_copy = []
    
    for operation in operations:
        operation_copy = operation.copy()
        
        # 如果是FillNaString操作，替换为CustomFillNaString
        if operation_copy.get('func_name') == 'FillNaString':
            operation_copy['func_name'] = 'CustomFillNaString'
            
        operations_copy.append(operation_copy)
    
    return operations_copy


def load_train_config(config_path: str = "config/train.yml") -> Dict[str, Any]:
    """
    加载训练配置
    
    Args:
        config_path: 配置文件路径，默认为config/train.yml
        
    Returns:
        训练配置字典
    
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML解析错误
    """
    return _load_yaml_config(config_path) 