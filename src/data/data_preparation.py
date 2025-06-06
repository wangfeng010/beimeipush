#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据准备工具模块，提供数据集构建和准备的功能
"""

import os
import datetime
import json
import glob
from typing import Dict, List, Tuple, Any, Optional

import tensorflow as tf
from src.data.dataset_utils import build_dataset, split_dataset


def prepare_datasets(data_config: Dict[str, Any], 
                     train_config: Dict[str, Any], 
                     tf_dtype_mapping: Dict[str, Any],
                     filter_column: Optional[str] = None):
    """
    准备训练和验证数据集
    
    参数:
        data_config: 数据配置字典
        train_config: 训练配置字典
        tf_dtype_mapping: 数据类型映射字典
        filter_column: 指定用于过滤的列名（只保留该列非空的数据）
    
    返回:
        full_dataset: 完整数据集
        train_dataset: 训练数据集
        validation_dataset: 验证数据集
        column_names: 列名列表
        input_signature: 输入签名
    """
    # 1. 查找数据文件
    csv_files = _find_data_files(data_config)
    
    # 2. 解析列信息
    column_names, column_defaults, column_types = _parse_column_config(data_config)
    
    # 3. 获取标签列
    label_columns = data_config.get('label_columns', ['log_type'])
    print(f"数据集列数: {len(column_names)}")
    print(f"标签列: {label_columns}")
    
    # 4. 显示过滤信息
    if filter_column:
        print(f"🔍 将对数据进行过滤，只保留 '{filter_column}' 列非空的数据")
    else:
        print("📊 使用全部数据进行训练")
    
    # 5. 构建原始数据集（添加过滤参数）
    dataset_with_userid, unique_user_ids, total_samples = build_dataset(
        _get_file_pattern(data_config), column_names, column_defaults, 
        data_config=data_config, filter_column=filter_column
    )
    
    # 6. 划分训练集和验证集
    train_dataset, validation_dataset, val_users, train_users = _split_train_val_dataset(
        dataset_with_userid, unique_user_ids, train_config
    )
    
    # 7. 配置批处理和预取
    train_dataset, validation_dataset, full_dataset = _configure_datasets(
        dataset_with_userid, train_dataset, validation_dataset, train_config
    )
    
    # 8. 记录用户划分信息
    log_user_split(train_users, val_users, unique_user_ids, total_samples, filter_column)
    
    # 9. 确定输入签名
    input_signature = _determine_input_signature(full_dataset)
    
    return full_dataset, train_dataset, validation_dataset, column_names, input_signature


def _find_data_files(data_config: Dict[str, Any]) -> List[str]:
    """
    查找数据文件
    
    参数:
        data_config: 数据配置字典
    
    返回:
        data_files: 数据文件路径列表
    """
    file_pattern = _get_file_pattern(data_config)
    
    # 查找数据文件
    data_files = glob.glob(file_pattern)
    
    # 确定文件类型用于显示
    file_type = "TXT" if _is_txt_format(data_config) else "CSV"
    
    print(f"找到{file_type}文件: {len(data_files)}个")
    print(f"示例文件: {data_files[:5] if data_files else '无'}")
    
    if not data_files:
        # 线上环境不应该创建测试数据，直接抛出错误
        raise FileNotFoundError(f"在 {data_config.get('train_dir', 'data/train')} 目录下未找到任何{file_type}文件。请检查数据路径和文件格式配置。")
    
    return data_files


def _get_file_pattern(data_config: Dict[str, Any]) -> str:
    """
    获取文件匹配模式
    
    参数:
        data_config: 数据配置字典
    
    返回:
        file_pattern: 文件匹配模式
    """
    train_dir = data_config.get('train_dir', 'data/train')
    
    # 根据配置确定文件扩展名
    if _is_txt_format(data_config):
        file_extension = "*.txt"
    else:
        file_extension = "*.csv"
    
    return os.path.join(train_dir, file_extension)


def _is_txt_format(data_config: Dict[str, Any]) -> bool:
    """
    判断是否使用TXT格式
    
    参数:
        data_config: 数据配置字典
    
    返回:
        bool: True表示使用TXT格式，False表示使用CSV格式
    """
    # 检查环境变量
    env_format = os.getenv('DATA_FORMAT', '').lower()
    if env_format == 'txt':
        return True
    elif env_format == 'csv':
        return False
    
    # 如果没有环境变量，检查配置文件中是否有txt_format且未被注释
    if data_config and 'txt_format' in data_config:
        return True
    
    return False


def _parse_column_config(data_config: Dict[str, Any]) -> Tuple[List[str], List[Any], Dict[str, str]]:
    """
    解析列配置信息
    
    参数:
        data_config: 数据配置字典
    
    返回:
        column_names: 列名列表
        column_defaults: 列默认值列表
        column_types: 列类型字典
    """
    raw_columns = data_config.get('raw_data_columns', [])
    column_names = []
    column_defaults = []
    column_types = {}
    
    # 解析列配置
    for col_item in raw_columns:
        for col_name, col_type in col_item.items():
            column_names.append(col_name)
            column_types[col_name] = col_type
            
            # 设置默认值
            if col_type == 'string':
                column_defaults.append('')
            else:
                column_defaults.append(0)
    
    return column_names, column_defaults, column_types


def _split_train_val_dataset(dataset_with_userid, 
                             unique_user_ids, 
                             train_config: Dict[str, Any]) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[Any], List[Any]]:
    """
    按用户ID划分训练集和验证集
    
    参数:
        dataset_with_userid: 带用户ID的数据集
        unique_user_ids: 唯一用户ID列表
        train_config: 训练配置字典
    
    返回:
        train_dataset: 训练数据集
        validation_dataset: 验证数据集
        val_users: 验证集用户ID列表
        train_users: 训练集用户ID列表
    """
    # 获取验证集比例
    val_ratio = train_config['training'].get('validation_split', 0.2) if train_config else 0.2
    
    # 按用户划分数据集
    train_dataset, validation_dataset, val_users, train_users = split_dataset(
        dataset_with_userid, unique_user_ids, val_ratio=val_ratio
    )
    
    return train_dataset, validation_dataset, val_users, train_users


def _configure_datasets(dataset_with_userid, 
                        train_dataset, 
                        validation_dataset, 
                        train_config: Dict[str, Any]) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    配置数据集的批处理、预取等参数
    
    参数:
        dataset_with_userid: 带用户ID的完整数据集
        train_dataset: 训练数据集
        validation_dataset: 验证数据集
        train_config: 训练配置字典
    
    返回:
        train_dataset: 配置后的训练数据集
        validation_dataset: 配置后的验证数据集
        full_dataset: 配置后的完整数据集
    """
    # 从训练配置获取批处理大小和shuffle buffer大小
    batch_size = train_config['training'].get('batch_size', 256) if train_config else 256
    shuffle_buffer_size = (
        train_config['training'].get('shuffle_buffer_size', 20000) 
        if train_config else 20000
    )
    
    print(f"批处理大小: {batch_size}, Shuffle缓冲区大小: {shuffle_buffer_size}")
    
    # 设置训练集批处理和shuffle
    configured_train_dataset = (
        train_dataset
        .shuffle(buffer_size=shuffle_buffer_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # 设置验证集批处理
    configured_validation_dataset = (
        validation_dataset
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # 创建一个完整的数据集用于整体分析
    full_dataset = (
        dataset_with_userid
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    return configured_train_dataset, configured_validation_dataset, full_dataset


def _determine_input_signature(dataset: tf.data.Dataset) -> Optional[Dict[str, tf.TensorShape]]:
    """
    确定数据集的输入签名
    
    参数:
        dataset: 数据集
    
    返回:
        input_signature: 输入签名字典
    """
    input_signature = None
    for batch in dataset.take(1):
        features, _ = batch
        input_signature = {k: v.shape for k, v in features.items()}
    return input_signature


def log_user_split(train_users, val_users, unique_user_ids, total_samples, filter_column):
    """
    记录用户划分信息到日志文件
    
    参数:
        train_users: 训练集用户ID列表
        val_users: 验证集用户ID列表
        unique_user_ids: 所有唯一用户ID列表
        total_samples: 总样本数
        filter_column: 过滤的列名
    """
    # 创建日志目录
    os.makedirs("./logs", exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    user_split_log = os.path.join("./logs", f"user_split_{timestamp}.json")
    
    # 计算比例
    train_ratio = 1.0 - (len(val_users) / len(unique_user_ids))
    val_ratio = len(val_users) / len(unique_user_ids)
    
    # 保存划分信息的摘要，避免文件过大
    split_info = {
        "train_users_count": len(train_users),
        "validation_users_count": len(val_users),
        "total_users_count": len(unique_user_ids),
        "total_samples": total_samples,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "filter_column": filter_column
    }
    
    with open(user_split_log, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"用户划分信息已保存到: {user_split_log}") 