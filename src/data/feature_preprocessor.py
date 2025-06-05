#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
特征预处理器
直接使用项目的操作函数实现特征预处理，将原始CSV数据转换为模型期望的格式
"""

import os
import sys
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Any
from functools import partial

# 导入项目本身的操作函数
from src.preprocess.operations import OP_HUB
from src.utils.config_utils import load_feature_config


def run_one_op_pd(df: pd.DataFrame, op) -> pd.DataFrame:
    """
    执行单个操作 - 使用项目的OP_HUB实现
    """
    col_in = op.col_in
    col_out = op.col_out
    func_name = op.func_name
    parameters = op.func_parameters if op.func_parameters else dict()
    partial_func = partial(OP_HUB[func_name], **parameters)

    if isinstance(col_in, list):
        df[col_out] = df[col_in].apply(lambda row: partial_func(*row), axis=1)
    else:
        df[col_out] = df[col_in].apply(partial_func)
    return df


def preprocess_features(df: pd.DataFrame, feat_configs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    使用项目的操作函数预处理特征
    
    Args:
        df: 输入DataFrame
        feat_configs: 特征配置列表
        
    Returns:
        处理后的DataFrame
    """
    working_df = df.copy()
    
    # 为每个特征执行操作链
    for config in feat_configs:
        operations = config.get('operations', [])
        
        # 执行操作链
        for op_dict in operations:
            # 创建操作对象
            from uniprocess.config import OperationConfig
            op = OperationConfig(**op_dict)
            
            # 检查输入列是否存在
            if isinstance(op.col_in, list):
                missing_cols = [col for col in op.col_in if col not in working_df.columns]
            else:
                missing_cols = [] if op.col_in in working_df.columns else [op.col_in]
            
            if missing_cols:
                print(f"WARNING: 缺失输入列 {missing_cols}，跳过操作 {op.func_name}")
                continue
            
            # 执行操作
            try:
                working_df = run_one_op_pd(working_df, op)
                print(f"✅ 执行操作: {op.col_in} --{op.func_name}--> {op.col_out}")
            except Exception as e:
                print(f"❌ 操作失败: {op.func_name}, 错误: {e}")
                continue
    
    return working_df


def apply_feature_preprocessing(dataset: tf.data.Dataset, 
                               feat_config_path: str = "config/feat.yml",
                               verbose: bool = True) -> tf.data.Dataset:
    """
    对TensorFlow数据集应用特征预处理
    
    Args:
        dataset: 原始数据集
        feat_config_path: 特征配置文件路径
        verbose: 是否输出详细日志
        
    Returns:
        处理后的数据集
    """
    if verbose:
        print("🔧 使用项目操作函数预处理特征...")
    
    # 加载特征配置
    feat_configs = load_feature_config(feat_config_path)
    
    # 将数据集转换为pandas DataFrame进行处理
    def process_pandas_batch(features_dict, labels_tensor):
        """使用pandas处理批次数据"""
        # 转换为pandas DataFrame
        batch_data = {}
        for name, tensor in features_dict.items():
            if tensor.dtype == tf.string:
                values = [item.decode('utf-8') if isinstance(item, bytes) else str(item) 
                         for item in tensor.numpy()]
            else:
                values = tensor.numpy().tolist()
            batch_data[name] = values
        
        df = pd.DataFrame(batch_data)
        
        # 使用项目操作函数处理
        processed_df = preprocess_features(df, feat_configs)
        
        # 创建处理后的特征字典
        processed_features = {}
        for config in feat_configs:
            feat_name = config['feat_name']
            if feat_name in processed_df.columns:
                feat_type = config.get('feat_type', 'sparse')
                values = processed_df[feat_name].tolist()
                
                if feat_type in ['sparse', 'varlen_sparse']:
                    if feat_type == 'varlen_sparse':
                        # 处理变长特征，确保所有列表长度一致
                        max_len = max(len(v) if isinstance(v, list) else 1 for v in values)
                        padded_values = []
                        for val in values:
                            if isinstance(val, list):
                                padded = val + [0] * (max_len - len(val))
                            else:
                                padded = [int(val) if val != 'null' else 0] + [0] * (max_len - 1)
                            padded_values.append(padded[:max_len])
                        processed_features[feat_name] = tf.constant(padded_values, dtype=tf.int32)
                    else:
                        # 单值特征
                        int_values = [int(x) if x != 'null' else 0 for x in values]
                        processed_features[feat_name] = tf.constant(int_values, dtype=tf.int32)
                elif feat_type == 'dense':
                    float_values = [float(x) for x in values]
                    processed_features[feat_name] = tf.constant(float_values, dtype=tf.float32)
                else:
                    str_values = [str(x) for x in values]
                    processed_features[feat_name] = tf.constant(str_values, dtype=tf.string)
        
        return processed_features, labels_tensor
    
    # 直接在eager模式下处理每个批次
    processed_batches = []
    
    if verbose:
        print("   开始批量处理数据...")
    
    batch_count = 0
    for features, labels in dataset:
        try:
            processed_features, processed_labels = process_pandas_batch(features, labels)
            processed_batches.append((processed_features, processed_labels))
            batch_count += 1
            if verbose and batch_count % 10 == 0:
                print(f"   已处理 {batch_count} 个批次")
        except Exception as e:
            if verbose:
                print(f"❌ 批次处理失败: {e}")
            raise e
    
    # 重新创建数据集
    def generator():
        for features, labels in processed_batches:
            yield features, labels
    
    # 创建输出签名
    feature_spec = {}
    for config in feat_configs:
        feat_name = config['feat_name']
        feat_type = config.get('feat_type', 'sparse')
        
        if feat_type == 'varlen_sparse':
            feature_spec[feat_name] = tf.TensorSpec(shape=(None, None), dtype=tf.int32)
        elif feat_type in ['sparse']:
            feature_spec[feat_name] = tf.TensorSpec(shape=(None,), dtype=tf.int32)
        elif feat_type == 'dense':
            feature_spec[feat_name] = tf.TensorSpec(shape=(None,), dtype=tf.float32)
        else:
            feature_spec[feat_name] = tf.TensorSpec(shape=(None,), dtype=tf.string)
    
    label_spec = tf.TensorSpec(shape=(None,), dtype=tf.int32)
    
    processed_dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(feature_spec, label_spec)
    )
    
    if verbose:
        print(f"✅ 特征预处理完成，共处理 {batch_count} 个批次")
    
    return processed_dataset 