#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置加载器模块，提供加载和处理配置文件的函数
"""

import os
import json
import yaml
import tensorflow as tf


def load_config_file(config_path):
    """
    加载配置文件，支持JSON和YAML格式
    
    参数:
        config_path: 配置文件路径
        
    返回:
        配置字典
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


def load_data_config(config_path="./config/data.yml"):
    """
    加载数据配置文件
    
    参数:
        config_path: 数据配置文件路径，默认为./config/data.yml
        
    返回:
        数据配置字典
    """
    try:
        return load_config_file(config_path)
    except Exception as e:
        print(f"加载数据配置失败: {e}")
        return {}


def load_train_config(config_path="./config/train.yml"):
    """
    加载训练配置文件
    
    参数:
        config_path: 训练配置文件路径，默认为./config/train.yml
        
    返回:
        训练配置字典
    """
    try:
        return load_config_file(config_path)
    except Exception as e:
        print(f"加载训练配置失败: {e}")
        return {}


def load_feature_config(config_path="./config/feat.yml", exclude_features=None):
    """
    加载特征配置文件，可以排除指定特征
    
    参数:
        config_path: 特征配置文件路径，默认为./config/feat.yml
        exclude_features: 要排除的特征列表，默认为None
        
    返回:
        处理后的特征配置列表
    """
    exclude_features = exclude_features or []
    
    try:
        config = load_config_file(config_path)
        
        # 检查配置格式并提取pipelines
        if 'pipelines' in config:
            pipelines = config['pipelines']
            
            # 排除指定特征
            if exclude_features:
                filtered_pipelines = []
                for pipeline in pipelines:
                    feat_name = pipeline.get('feat_name', '')
                    if feat_name not in exclude_features:
                        filtered_pipelines.append(pipeline)
                return filtered_pipelines
            
            return pipelines
        else:
            print(f"警告: 特征配置中未找到'pipelines'键")
            return []
    except Exception as e:
        print(f"加载特征配置失败: {e}")
        return []


def extract_config_info(data_config, dtype_mapping):
    """
    从数据配置中提取信息
    
    参数:
        data_config: 数据配置字典
        dtype_mapping: 数据类型映射字典
        
    返回:
        file_pattern: 文件模式
        column_names: 列名列表
        column_defaults: 列默认值列表
        label_columns: 标签列名列表
    """
    file_pattern = data_config.get('file_pattern', '')
    column_names = data_config.get('column_names', [])
    column_defaults = []
    
    # 构建列默认值
    for col_name in column_names:
        col_type = data_config.get('column_types', {}).get(col_name, 'float32')
        if col_type in dtype_mapping:
            dtype = dtype_mapping[col_type]
            default_value = '' if dtype == tf.string else 0
            column_defaults.append(default_value)
        else:
            column_defaults.append(0)  # 默认为数值0
    
    label_columns = data_config.get('label_columns', ['label'])
    
    return file_pattern, column_names, column_defaults, label_columns 