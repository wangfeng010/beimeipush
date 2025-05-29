#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset utilities for push binary classification
"""

import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Set, Any


def build_dataset(file_pattern: str, column_names: List[str], column_defaults: List[Any], 
                  batch_size: int = 256, data_config: Dict = None) -> Tuple[tf.data.Dataset, np.ndarray, int]:
    """
    构建数据集
    
    Args:
        file_pattern: 文件匹配模式
        column_names: 数据列名
        column_defaults: 每列的默认数据类型
        batch_size: 批处理大小
        data_config: 数据配置字典，包含CSV格式设置
        
    Returns:
        dataset: TensorFlow 数据集
        unique_user_ids: 唯一用户ID数组
        total_samples: 样本总数
    """
    # 添加调试信息
    files = glob.glob(file_pattern)
    print(f"找到CSV文件: {len(files)}个")
    print(f"示例文件: {files[:3]}")
    
    if not files:
        raise ValueError(f"在 {file_pattern} 找不到任何CSV文件")
    
    # 使用pandas读取所有CSV文件
    print("读取CSV文件...")
    dfs = []
    
    # 显式指定每列的类型，确保一致性
    dtypes = {
        'user_id': str,
        'create_time': str,
        'log_type': str,
        'watchlists': str, 
        'holdings': str,
        'country': str,
        'prefer_bid': str,
        'user_propernoun': str,
        'push_title': str,
        'push_content': str,
        'item_code': str,
        'item_tags': str,
        'submit_type': str
    }
    
    # 从配置中获取CSV格式设置
    csv_sep = ','  # 默认逗号分隔
    csv_header = 0  # 默认有表头
    
    if data_config and 'csv_format' in data_config:
        csv_format = data_config['csv_format']
        csv_sep = csv_format.get('separator', ',')
        csv_header = csv_format.get('header', 0)
        print(f"使用配置的CSV格式: 分隔符='{csv_sep}', 表头={csv_header}")
    
    # 决定是否需要提供列名
    use_names = None
    if csv_header is None:  # 如果无表头，则使用提供的列名
        use_names = column_names
    
    for file in files:
        print(f"读取文件: {file}")
        try:
            df = pd.read_csv(file, sep=csv_sep, header=csv_header, names=use_names,
                           escapechar='\\', quotechar='"', dtype=dtypes)
            print(f"  行数: {len(df)}")
            dfs.append(df)
        except Exception as e:
            print(f"  读取错误: {e}")
    
    if not dfs:
        raise ValueError("没有成功读取任何CSV文件")
        
    # 合并所有数据框
    combined_df = pd.concat(dfs)
    print(f"合并后总行数: {len(combined_df)}")
    
    # 显示数据类型
    print("数据列和类型:")
    print(combined_df.dtypes)
    
    # 检查标签列的值分布
    print("标签值分布:")
    print(combined_df['log_type'].value_counts())
    
    # 提取标签列
    labels = combined_df.pop('log_type')
    
    # 将标签转换为0/1 (PR:0, PC:1)
    label_dict = {'PR': 0, 'PC': 1}
    numeric_labels = labels.map(lambda x: label_dict.get(x, -1))
    print("标签数值分布:")
    print(numeric_labels.value_counts())
    
    # 保存user_id列以便后续按用户划分训练和测试集
    user_ids = combined_df['user_id'].values
    
    # 转换为TensorFlow数据集
    features_dict = {}
    
    # 处理特征，保持原始数据类型，处理空值
    for col in combined_df.columns:
        # 将NaN值替换为空字符串
        combined_df[col] = combined_df[col].fillna('')
        features_dict[col] = tf.constant(combined_df[col].values, dtype=tf.string)
    
    # 创建数据集，包含特征和标签
    labels_tensor = tf.constant(numeric_labels.values, dtype=tf.int32)
    
    # 创建包含所有特征的数据集
    dataset = tf.data.Dataset.from_tensor_slices((features_dict, labels_tensor))
    
    # 返回数据集、唯一用户ID和总样本数
    unique_user_ids = np.unique(user_ids)
    total_samples = len(combined_df)
    
    return dataset, unique_user_ids, total_samples


def split_dataset(dataset: tf.data.Dataset, unique_user_ids: np.ndarray, 
                 val_ratio: float = 0.2) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str], List[str]]:
    """
    按用户划分数据集为训练集和验证集
    
    Args:
        dataset: 包含用户ID的数据集
        unique_user_ids: 唯一用户ID数组
        val_ratio: 验证集比例
        
    Returns:
        train_dataset: 训练数据集
        validation_dataset: 验证数据集
        val_users: 验证集用户ID列表
        train_users: 训练集用户ID列表
    """
    print(f"按用户拆分数据集，验证集比例: {val_ratio}")
    print(f"唯一用户数: {len(unique_user_ids)}")
    
    # 随机打乱用户ID
    np.random.shuffle(unique_user_ids)
    
    # 按比例划分用户为训练集和验证集
    split_idx = int(len(unique_user_ids) * (1 - val_ratio))
    train_users = unique_user_ids[:split_idx]
    val_users = unique_user_ids[split_idx:]
    
    print(f"训练集用户数: {len(train_users)}")
    print(f"验证集用户数: {len(val_users)}")
    
    # 将用户ID转换为集合以便快速查找
    val_users_set = set(val_users)
    
    # 函数：确定某条记录属于训练集还是验证集
    def is_validation_user(features, label):
        return tf.py_function(
            lambda x: tf.constant(x.numpy().decode('utf-8') in val_users_set), 
            [features["user_id"]], 
            tf.bool
        )
    
    # 使用filter函数按用户ID划分数据集
    validation_dataset = dataset.filter(is_validation_user)
    train_dataset = dataset.filter(
        lambda features, label: tf.logical_not(is_validation_user(features, label))
    )
    
    # 验证数据集大小
    try:
        # 获取大致的批次数量
        train_count = sum(1 for _ in train_dataset.take(100))
        val_count = sum(1 for _ in validation_dataset.take(100))
        
        print(f"训练集样本数(估计): 至少 {train_count} 条")
        print(f"验证集样本数(估计): 至少 {val_count} 条")
    except Exception as e:
        print(f"计算数据集大小时出错: {e}")
    
    return train_dataset, validation_dataset, val_users.tolist(), train_users.tolist()


def validate_dataset(dataset: tf.data.Dataset) -> Dict[str, tf.TensorSpec]:
    """
    验证数据集并返回输入签名
    
    Args:
        dataset: 要验证的数据集
        
    Returns:
        input_signature: 输入签名字典
    """
    print("验证数据集...")
    
    # 获取第一个批次
    for features_batch, labels_batch in dataset.take(1):
        print(f"特征批次类型: {type(features_batch)}")
        print(f"标签批次类型: {type(labels_batch)}")
        print(f"标签批次形状: {labels_batch.shape}")
        
        # 检查特征数据的结构
        print("\n特征数据结构:")
        for key, tensor in features_batch.items():
            print(f"  {key}: 类型={tensor.dtype}, 形状={tensor.shape}")
        
        # 创建输入签名字典
        input_signature = {}
        for key, tensor in features_batch.items():
            input_signature[key] = tf.TensorSpec(shape=[None], dtype=tensor.dtype, name=key)
        
        return input_signature
    
    raise ValueError("数据集为空，无法验证")


def inspect_datasets(dataset: tf.data.Dataset, train_dataset: tf.data.Dataset, 
                    validation_dataset: tf.data.Dataset) -> None:
    """
    检查数据集的样本和分布情况
    
    Args:
        dataset: 完整数据集
        train_dataset: 训练数据集
        validation_dataset: 验证数据集
    """
    print("\n=== 数据集样本检查 ===")
    
    # 检查一个批次的特征和标签
    for features, labels in dataset.take(1):
        print("样本特征:")
        for name, values in features.items():
            print(f"  {name}: 形状={values.shape}, 类型={values.dtype}")
            # 显示第一个样本的值
            if values.shape[0] > 0:
                try:
                    if values.dtype == tf.string:
                        print(f"    样本值: {values[0].numpy().decode('utf-8')}")
                    else:
                        print(f"    样本值: {values[0].numpy()}")
                except:
                    pass
        
        print(f"标签形状: {labels.shape}")
        if labels.shape[0] > 0:
            print(f"  样本标签: {labels[0].numpy()}")
    
    # 检查训练集和验证集的批次大小
    print("\n训练集和验证集批次大小:")
    for features, _ in train_dataset.take(1):
        print(f"训练数据集批次大小: {next(iter(features.values())).shape[0]}")
    
    for features, _ in validation_dataset.take(1):
        print(f"验证数据集批次大小: {next(iter(features.values())).shape[0]}")
    
    print("\n数据集验证完成") 