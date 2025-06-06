#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset utilities for push binary classification
"""

import os
import glob
import json
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Set, Any, Optional


def build_dataset(
    file_pattern: str, 
    column_names: List[str], 
    column_defaults: List[Any], 
    batch_size: int = 256, 
    data_config: Optional[Dict] = None,
    filter_column: Optional[str] = None
) -> Tuple[tf.data.Dataset, np.ndarray, int]:
    """
    构建数据集
    
    Args:
        file_pattern: 文件匹配模式
        column_names: 数据列名
        column_defaults: 每列的默认数据类型
        batch_size: 批处理大小
        data_config: 数据配置字典，包含CSV格式设置
        filter_column: 指定用于过滤的列名（只保留该列非空的数据）
        
    Returns:
        dataset: TensorFlow 数据集
        unique_user_ids: 唯一用户ID数组
        total_samples: 样本总数
    """
    # 查找符合模式的文件
    files = _find_csv_files(file_pattern)
    
    # 从配置中获取CSV格式设置
    csv_sep, csv_header, use_names = _get_csv_format_settings(data_config, column_names)
    
    # 读取并合并CSV文件
    combined_df = _read_and_combine_data_files(files, csv_sep, csv_header, use_names)
    
    # 应用数据过滤（如果指定了过滤列）
    if filter_column:
        combined_df = _filter_data_by_column(combined_df, filter_column)
    
    # 显示数据信息
    _display_dataframe_info(combined_df)
    
    # 处理标签列
    numeric_labels = _process_labels(combined_df)
    
    # 保存user_id列以便后续按用户划分训练和测试集
    user_ids = combined_df['user_id'].values
    
    # 将数据转换为TensorFlow数据集
    dataset = _convert_to_tf_dataset(combined_df, numeric_labels)
    
    # 返回数据集、唯一用户ID和总样本数
    unique_user_ids = np.unique(user_ids)
    total_samples = len(combined_df)
    
    return dataset, unique_user_ids, total_samples


def _find_csv_files(file_pattern: str) -> List[str]:
    """查找符合模式的数据文件"""
    files = glob.glob(file_pattern)
    # 根据文件扩展名确定文件类型
    file_type = "TXT" if ".txt" in file_pattern else "CSV"
    print(f"找到{file_type}文件: {len(files)}个")
    if len(files) > 3:
        print(f"示例文件: {files[:5]}")
    else:
        print(f"示例文件: {files}")
    
    if not files:
        raise ValueError(f"在 {file_pattern} 找不到任何{file_type}文件")
    
    return files


def _get_csv_format_settings(
    data_config: Optional[Dict], 
    column_names: List[str]
) -> Tuple[str, Optional[int], Optional[List[str]]]:
    """从配置中获取数据格式设置（支持csv_format和txt_format）"""
    csv_sep = ','  # 默认逗号分隔
    csv_header = 0  # 默认有表头
    
    # 检查环境变量来决定使用哪种格式
    env_format = os.getenv('DATA_FORMAT', '').lower()  # 'txt' 或 'csv'
    
    # 如果设置了环境变量，优先使用指定格式
    if env_format == 'txt' and data_config and 'txt_format' in data_config:
        txt_format = data_config['txt_format']
        csv_sep = txt_format.get('separator', '\t')
        csv_header = txt_format.get('header', None)
        print(f"🌍 环境变量指定使用TXT格式: 分隔符='{csv_sep}', 表头={csv_header}")
    elif env_format == 'csv' and data_config and 'csv_format' in data_config:
        csv_format = data_config['csv_format']
        csv_sep = csv_format.get('separator', ',')
        csv_header = csv_format.get('header', 0)
        print(f"🌍 环境变量指定使用CSV格式: 分隔符='{csv_sep}', 表头={csv_header}")
    # 如果没有设置环境变量，使用原来的优先级逻辑
    elif data_config and 'txt_format' in data_config:
        txt_format = data_config['txt_format']
        csv_sep = txt_format.get('separator', '\t')
        csv_header = txt_format.get('header', None)
        print(f"使用线上TXT格式: 分隔符='{csv_sep}', 表头={csv_header}")
    elif data_config and 'csv_format' in data_config:
        csv_format = data_config['csv_format']
        csv_sep = csv_format.get('separator', ',')
        csv_header = csv_format.get('header', 0)
        print(f"使用本地CSV格式: 分隔符='{csv_sep}', 表头={csv_header}")
    else:
        print(f"使用默认格式: 分隔符='{csv_sep}', 表头={csv_header}")
    
    # 决定是否需要提供列名
    use_names = None
    if csv_header is None:  # 如果无表头，则使用提供的列名
        use_names = column_names
    
    return csv_sep, csv_header, use_names


def _read_and_combine_data_files(
    files: List[str], 
    csv_sep: str, 
    csv_header: Optional[int], 
    use_names: Optional[List[str]]
) -> pd.DataFrame:
    """读取并合并数据文件（支持CSV和TXT格式）"""
    # 根据文件扩展名确定文件类型
    file_type = "TXT" if any(".txt" in f for f in files) else "CSV"
    print(f"读取{file_type}文件...")
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
    
    for file in files:
        print(f"读取文件: {file}")
        try:
            df = pd.read_csv(
                file, 
                sep=csv_sep, 
                header=csv_header, 
                names=use_names,
                escapechar='\\', 
                quotechar='"', 
                dtype=dtypes
            )
            print(f"  行数: {len(df)}")
            dfs.append(df)
        except Exception as e:
            print(f"  读取错误: {e}")
    
    if not dfs:
        raise ValueError(f"没有成功读取任何{file_type}文件")
        
    # 合并所有数据框
    combined_df = pd.concat(dfs)
    print(f"合并后总行数: {len(combined_df)}")
    
    return combined_df


def _display_dataframe_info(df: pd.DataFrame) -> None:
    """显示数据框的基本信息"""
    print("数据列和类型:")
    print(df.dtypes)
    
    # 检查标签列的值分布
    print("标签值分布:")
    print(df['log_type'].value_counts())


def _process_labels(df: pd.DataFrame) -> pd.Series:
    """处理标签列"""
    # 提取标签列
    labels = df.pop('log_type')
    
    # 将标签转换为0/1 (PR:0, PC:1)
    label_dict = {'PR': 0, 'PC': 1}
    numeric_labels = labels.map(lambda x: label_dict.get(x, -1))
    print("标签数值分布:")
    print(numeric_labels.value_counts())
    
    return numeric_labels


def _convert_to_tf_dataset(df: pd.DataFrame, labels: pd.Series) -> tf.data.Dataset:
    """将DataFrame和标签转换为TensorFlow数据集"""
    features_dict = {}
    
    # 处理特征，保持原始数据类型，处理空值
    for col in df.columns:
        # 将NaN值替换为空字符串
        df[col] = df[col].fillna('')
        features_dict[col] = tf.constant(df[col].values, dtype=tf.string)
    
    # 创建数据集，包含特征和标签
    labels_tensor = tf.constant(labels.values, dtype=tf.int32)
    
    # 创建包含所有特征的数据集
    return tf.data.Dataset.from_tensor_slices((features_dict, labels_tensor))


def split_dataset(
    dataset: tf.data.Dataset, 
    unique_user_ids: np.ndarray, 
    val_ratio: float = 0.2
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str], List[str]]:
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
    
    # 获取训练和验证用户列表
    train_users, val_users = _split_users(unique_user_ids, val_ratio)
    
    # 将用户ID转换为集合以便快速查找
    val_users_set = set(val_users)
    
    # 根据用户ID划分数据集
    train_dataset, validation_dataset = _filter_datasets_by_users(dataset, val_users_set)
    
    # 验证数据集大小
    _validate_split_sizes(train_dataset, validation_dataset)
    
    # 保存用户划分信息到日志
    _save_user_split_info(train_users, val_users)
    
    return train_dataset, validation_dataset, val_users.tolist(), train_users.tolist()


def _split_users(user_ids: np.ndarray, val_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """将用户ID拆分为训练和验证集"""
    # 随机打乱用户ID
    np.random.shuffle(user_ids)
    
    # 按比例划分用户为训练集和验证集
    split_idx = int(len(user_ids) * (1 - val_ratio))
    train_users = user_ids[:split_idx]
    val_users = user_ids[split_idx:]
    
    print(f"训练集用户数: {len(train_users)}")
    print(f"验证集用户数: {len(val_users)}")
    
    return train_users, val_users


def _filter_datasets_by_users(
    dataset: tf.data.Dataset, 
    val_users_set: Set[str]
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """根据用户ID过滤数据集"""
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
    
    return train_dataset, validation_dataset


def _validate_split_sizes(
    train_dataset: tf.data.Dataset, 
    validation_dataset: tf.data.Dataset
) -> None:
    """验证拆分后的数据集大小"""
    try:
        # 获取大致的批次数量
        train_count = sum(1 for _ in train_dataset.take(100))
        val_count = sum(1 for _ in validation_dataset.take(100))
        
        print(f"训练集样本数(估计): 至少 {train_count} 条")
        print(f"验证集样本数(估计): 至少 {val_count} 条")
    except Exception as e:
        print(f"计算数据集大小时出错: {e}")


def _save_user_split_info(train_users: np.ndarray, val_users: np.ndarray) -> None:
    """保存用户划分信息到日志文件"""
    try:
        # 确保日志目录存在
        os.makedirs("./logs", exist_ok=True)
        
        # 保存用户划分信息
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        split_info = {
            "train_users_count": len(train_users),
            "val_users_count": len(val_users),
            "timestamp": timestamp
        }
        
        # 写入JSON文件
        split_file = f"./logs/user_split_{timestamp}.json"
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"用户划分信息已保存到: {split_file}")
    except Exception as e:
        print(f"保存用户划分信息时出错: {e}")


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


def inspect_datasets(
    dataset: tf.data.Dataset, 
    train_dataset: tf.data.Dataset, 
    validation_dataset: tf.data.Dataset
) -> None:
    """
    检查数据集的样本和分布情况
    
    Args:
        dataset: 完整数据集
        train_dataset: 训练数据集
        validation_dataset: 验证数据集
    """
    print("\n=== 数据集样本检查 ===")
    
    # 检查一个批次的特征和标签
    _inspect_batch_features_and_labels(dataset)
    
    # 检查训练集和验证集的批次大小
    _inspect_batch_sizes(train_dataset, validation_dataset)
    
    print("\n数据集验证完成")


def _inspect_batch_features_and_labels(dataset: tf.data.Dataset) -> None:
    """检查数据批次的特征和标签"""
    for features, labels in dataset.take(1):
        # 检查特征
        _inspect_features(features)
        
        # 检查标签
        _inspect_labels(labels)
        
        # 只处理第一个批次
        break


def _inspect_features(features: Dict[str, tf.Tensor]) -> None:
    """检查并打印特征信息"""
    print("样本特征:")
    for name, values in features.items():
        print(f"  {name}: 形状={values.shape}, 类型={values.dtype}")
        
        # 尝试显示第一个样本的值
        if values.shape[0] > 0:
            _print_sample_value(name, values)


def _print_sample_value(name: str, values: tf.Tensor) -> None:
    """打印特征的样本值"""
    try:
        if values.dtype == tf.string:
            sample_value = values[0].numpy().decode('utf-8')
            # 如果样本值过长，截断显示
            if len(sample_value) > 100:
                sample_value = sample_value[:100] + "..."
            print(f"    样本值: {sample_value}")
        else:
            print(f"    样本值: {values[0].numpy()}")
    except Exception:
        pass  # 忽略无法显示的值


def _inspect_labels(labels: tf.Tensor) -> None:
    """检查并打印标签信息"""
    print(f"标签形状: {labels.shape}")
    if labels.shape[0] > 0:
        print(f"  样本标签: {labels[0].numpy()}")


def _inspect_batch_sizes(
    train_dataset: tf.data.Dataset, 
    validation_dataset: tf.data.Dataset
) -> None:
    """检查训练集和验证集的批次大小"""
    print("\n训练集和验证集批次大小:")
    
    # 检查训练集批次大小
    for features, _ in train_dataset.take(1):
        batch_size = next(iter(features.values())).shape[0]
        print(f"训练数据集批次大小: {batch_size}")
        break
    
    # 检查验证集批次大小
    for features, _ in validation_dataset.take(1):
        batch_size = next(iter(features.values())).shape[0]
        print(f"验证数据集批次大小: {batch_size}")
        break 


def _filter_data_by_column(df: pd.DataFrame, filter_column: str) -> pd.DataFrame:
    """
    根据指定列过滤数据，只保留该列非空的记录
    
    Args:
        df: 原始数据框
        filter_column: 用于过滤的列名
        
    Returns:
        filtered_df: 过滤后的数据框
    """
    print(f"\n🔍 开始根据 '{filter_column}' 列过滤数据...")
    
    # 检查列是否存在
    if filter_column not in df.columns:
        print(f"⚠️  警告: 列 '{filter_column}' 不存在于数据中，跳过过滤")
        return df
    
    # 显示过滤前的数据统计
    total_before = len(df)
    null_count = df[filter_column].isna().sum() + (df[filter_column] == '').sum()
    non_null_count = total_before - null_count
    
    print(f"过滤前数据统计:")
    print(f"  总样本数: {total_before:,}")
    print(f"  {filter_column} 为空的样本: {null_count:,} ({null_count/total_before*100:.1f}%)")
    print(f"  {filter_column} 非空的样本: {non_null_count:,} ({non_null_count/total_before*100:.1f}%)")
    
    # 执行过滤：保留非空且非空字符串的记录
    filtered_df = df[(df[filter_column].notna()) & (df[filter_column] != '')]
    
    # 显示过滤后的数据统计
    total_after = len(filtered_df)
    reduction_rate = (total_before - total_after) / total_before * 100
    
    print(f"过滤后数据统计:")
    print(f"  保留样本数: {total_after:,}")
    print(f"  过滤掉样本数: {total_before - total_after:,}")
    print(f"  数据减少比例: {reduction_rate:.1f}%")
    
    # 检查过滤后的标签分布
    if 'log_type' in filtered_df.columns:
        print(f"过滤后标签分布:")
        print(filtered_df['log_type'].value_counts())
    
    return filtered_df 