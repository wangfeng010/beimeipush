#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据准备工具模块，提供数据集构建和准备的功能
"""

import os
import datetime
import json
import glob
import tensorflow as tf
from src.data.dataset_utils import build_dataset, split_dataset


def prepare_datasets(data_config, train_config, tf_dtype_mapping):
    """
    准备训练和验证数据集
    
    参数:
        data_config: 数据配置字典
        train_config: 训练配置字典
        tf_dtype_mapping: 数据类型映射字典
    
    返回:
        full_dataset: 完整数据集
        train_dataset: 训练数据集
        validation_dataset: 验证数据集
        column_names: 列名列表
        input_signature: 输入签名
    """
    # 从配置中提取路径
    train_dir = data_config.get('train_dir', 'data/train')
    
    # 构建文件模式
    file_pattern = os.path.join(train_dir, "*.csv")
    
    # 检查文件是否存在
    csv_files = glob.glob(file_pattern)
    print(f"找到CSV文件: {len(csv_files)}个")
    print(f"示例文件: {csv_files[:5]}")
    
    if not csv_files:
        # 创建测试数据以便继续
        print("未找到实际数据文件，将创建一个小的测试数据集进行测试")
        create_test_data(train_dir)
        csv_files = glob.glob(file_pattern)
    
    # 获取数据列信息
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
    
    # 获取标签列
    label_columns = data_config.get('label_columns', ['log_type'])
    
    print(f"数据集列数: {len(column_names)}")
    print(f"标签列: {label_columns}")
    
    # 构建数据集
    dataset_with_userid, unique_user_ids, total_samples = build_dataset(
        file_pattern, column_names, column_defaults
    )
    
    # 获取验证集比例
    val_ratio = train_config['training'].get('validation_split', 0.2) if train_config else 0.2
    
    # 按用户划分数据集
    train_dataset, validation_dataset, val_users, train_users = split_dataset(
        dataset_with_userid, unique_user_ids, val_ratio=val_ratio
    )
    
    # 从训练配置获取批处理大小和shuffle buffer大小
    batch_size = train_config['training'].get('batch_size', 256) if train_config else 256
    shuffle_buffer_size = train_config['training'].get('shuffle_buffer_size', 20000) if train_config else 20000
    
    print(f"批处理大小: {batch_size}, Shuffle缓冲区大小: {shuffle_buffer_size}")
    
    # 设置训练集批处理和shuffle
    train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 设置验证集批处理
    validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 创建一个完整的数据集用于整体分析
    full_dataset = dataset_with_userid.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 记录用户划分信息
    log_user_split(train_users, val_users, unique_user_ids, total_samples)
    
    # 确定输入签名
    input_signature = None
    for batch in full_dataset.take(1):
        features, _ = batch
        input_signature = {k: v.shape for k, v in features.items()}
    
    return full_dataset, train_dataset, validation_dataset, column_names, input_signature


def log_user_split(train_users, val_users, unique_user_ids, total_samples):
    """
    记录用户划分信息到日志文件
    
    参数:
        train_users: 训练集用户ID列表
        val_users: 验证集用户ID列表
        unique_user_ids: 所有唯一用户ID列表
        total_samples: 总样本数
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    user_split_log = os.path.join("./logs", f"user_split_{timestamp}.json")
    
    # 保存划分信息的摘要，避免文件过大
    with open(user_split_log, 'w') as f:
        json.dump({
            "train_users_count": len(train_users),
            "validation_users_count": len(val_users),
            "total_users_count": len(unique_user_ids),
            "total_samples": total_samples,
            "train_ratio": 1.0 - (len(val_users) / len(unique_user_ids)),
            "val_ratio": len(val_users) / len(unique_user_ids)
        }, f, indent=2)
    print(f"用户划分信息已保存到: {user_split_log}")


def create_test_data(data_dir):
    """
    创建测试数据集用于调试
    
    参数:
        data_dir: 数据目录
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建一个小的测试CSV文件
    test_file = os.path.join(data_dir, "test_data.csv")
    
    # 简单的CSV头和数据
    header = "user_id,create_time,log_type,watchlists,holdings,country,prefer_bid,user_propernoun,push_title,push_content,item_code,item_tags,submit_type"
    
    # 生成10条测试数据
    rows = []
    for i in range(10):
        user_id = f"user_{i}"
        create_time = "2023-01-01 12:00:00"
        log_type = "1" if i % 2 == 0 else "0"  # 一半是正样本，一半是负样本
        row = f"{user_id},{create_time},{log_type},list1 & list2,,CN,bid1#1|bid2#2,,Test Title,Test Content,code123,tag1 tag2,"
        rows.append(row)
    
    # 写入文件
    with open(test_file, 'w') as f:
        f.write(header + "\n")
        f.write("\n".join(rows))
    
    print(f"已创建测试数据文件: {test_file}")
    print(f"测试数据包含 {len(rows)} 条记录") 