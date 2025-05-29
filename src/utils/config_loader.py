#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置加载器模块，提供加载和处理配置文件的函数
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Tuple, Any, Optional, Union

import tensorflow as tf


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件，支持JSON和YAML格式
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
        
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 不支持的文件格式
        IOError: 文件读取或解析错误
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    _, ext = os.path.splitext(config_path)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if ext.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif ext.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {ext}")
    except Exception as e:
        raise IOError(f"加载配置文件失败: {e}")


def load_data_config(config_path: str = "./config/data.yml") -> Dict[str, Any]:
    """
    加载数据配置文件
    
    Args:
        config_path: 数据配置文件路径，默认为./config/data.yml
        
    Returns:
        数据配置字典，如果加载失败则返回空字典
    """
    try:
        config = load_config_file(config_path)
        return config
    except Exception as e:
        print(f"加载数据配置失败: {e}")
        return {}


def load_train_config(config_path: str = "./config/train.yml") -> Dict[str, Any]:
    """
    加载训练配置文件
    
    Args:
        config_path: 训练配置文件路径，默认为./config/train.yml
        
    Returns:
        训练配置字典，如果加载失败则返回空字典
    """
    try:
        config = load_config_file(config_path)
        return config
    except Exception as e:
        print(f"加载训练配置失败: {e}")
        return {}


def load_feature_config(
    config_path: str = "./config/feat.yml", 
    exclude_features: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    加载特征配置文件，可以排除指定特征
    
    Args:
        config_path: 特征配置文件路径，默认为./config/feat.yml
        exclude_features: 要排除的特征列表，默认为None
        
    Returns:
        处理后的特征配置列表，如果加载失败或没有pipelines则返回空列表
    """
    # 将None转为空列表
    exclude_features = exclude_features or []
    
    try:
        # 加载配置文件
        config = load_config_file(config_path)
        
        # 获取pipelines并处理
        return _filter_feature_pipelines(config, exclude_features)
    except Exception as e:
        print(f"加载特征配置失败: {e}")
        return []


def _filter_feature_pipelines(
    config: Dict[str, Any], 
    exclude_features: List[str]
) -> List[Dict[str, Any]]:
    """
    过滤特征管道配置
    
    Args:
        config: 配置字典
        exclude_features: 要排除的特征列表
    
    Returns:
        过滤后的管道配置列表
    """
    # 检查配置中是否包含pipelines
    if 'pipelines' not in config:
        print("警告: 特征配置中未找到'pipelines'键")
        return []
    
    pipelines = config['pipelines']
    
    # 如果没有需要排除的特征，直接返回所有管道
    if not exclude_features:
        return pipelines
    
    # 过滤掉被排除的特征
    filtered_pipelines = []
    for pipeline in pipelines:
        feat_name = pipeline.get('feat_name', '')
        if feat_name and feat_name not in exclude_features:
            filtered_pipelines.append(pipeline)
    
    return filtered_pipelines


def extract_config_info(
    data_config: Dict[str, Any], 
    dtype_mapping: Dict[str, tf.DType]
) -> Tuple[str, List[str], List[Any], List[str]]:
    """
    从数据配置中提取信息
    
    Args:
        data_config: 数据配置字典
        dtype_mapping: 数据类型映射字典
        
    Returns:
        包含以下元素的元组:
            - file_pattern: 文件模式
            - column_names: 列名列表
            - column_defaults: 列默认值列表
            - label_columns: 标签列名列表
    """
    # 提取基本配置
    file_pattern = data_config.get('file_pattern', '')
    column_names = data_config.get('column_names', [])
    column_defaults = _build_column_defaults(data_config, column_names, dtype_mapping)
    label_columns = data_config.get('label_columns', ['label'])
    
    return file_pattern, column_names, column_defaults, label_columns


def _build_column_defaults(
    data_config: Dict[str, Any], 
    column_names: List[str], 
    dtype_mapping: Dict[str, tf.DType]
) -> List[Any]:
    """
    构建列默认值列表
    
    Args:
        data_config: 数据配置字典
        column_names: 列名列表
        dtype_mapping: 数据类型映射字典
    
    Returns:
        列默认值列表
    """
    column_defaults = []
    column_types = data_config.get('column_types', {})
    
    for col_name in column_names:
        # 获取列的数据类型，默认为float32
        col_type = column_types.get(col_name, 'float32')
        
        # 根据数据类型确定默认值
        if col_type in dtype_mapping:
            dtype = dtype_mapping[col_type]
            default_value = '' if dtype == tf.string else 0
            column_defaults.append(default_value)
        else:
            # 未知类型默认为数值0
            column_defaults.append(0)
    
    return column_defaults 