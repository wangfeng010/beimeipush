#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLP推送二分类模型训练脚本
"""

import os
import sys
# 将项目根目录和env目录添加到Python路径
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取项目根目录
sys.path.insert(0, project_dir)  # 将项目根目录添加到Python路径
env_dir = os.path.join(project_dir, 'env')
sys.path.insert(0, env_dir)  # 将env目录添加到Python路径
print(f"已添加到Python路径: {project_dir}")
print(f"已添加到Python路径: {env_dir}")

import numpy as np
import tensorflow as tf
import datetime
import json

# 设置GPU内存增长，避免大数据集训练时内存溢出
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"找到 {len(gpus)} 个GPU设备，已启用内存增长")
    except RuntimeError as e:
        print(f"GPU配置错误: {e}")

# 导入自定义模块
from light_ctr.utils.constants import TF_DTYPE_MAPPING
from src.utils.train_utils import (
    setup_environment, 
    load_data_config, 
    extract_config_info, 
    load_feature_config,
    load_train_config, 
    train_model, 
    check_feature_importance,
    plot_feature_importance
)
from src.data.dataset_utils import (
    build_dataset, 
    split_dataset, 
    validate_dataset, 
    inspect_datasets
)
from src.models.mlp import MLP


def main():
    """主函数，执行训练流程"""
    # 设置环境
    project_root = setup_environment()
    
    # 确保日志和模型目录存在
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    
    # 设置随机种子以确保结果可复现
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # 加载配置
    data_config = load_data_config()
    print("数据配置:")
    print(data_config)
    
    # 加载训练配置
    try:
        train_config = load_train_config()
        print("已加载训练配置:")
        print(train_config)
    except Exception as e:
        print(f"加载训练配置失败: {e}，将使用默认值")
        train_config = None
    
    # 提取配置信息
    file_pattern, column_names, column_defaults, label_columns = extract_config_info(data_config, TF_DTYPE_MAPPING)
    print("列名:", column_names)
    print("列默认值:", column_defaults)
    
    # 构建数据集
    dataset_with_userid, unique_user_ids, total_samples = build_dataset(
        file_pattern, column_names, column_defaults
    )
    
    # 按用户划分数据集
    train_dataset, validation_dataset, val_users, train_users = split_dataset(
        dataset_with_userid, unique_user_ids, 
        val_ratio=train_config['training'].get('validation_split', 0.2) if train_config else 0.2
    )
    
    # 从训练配置获取批处理大小和shuffle buffer大小
    if train_config is None:
        batch_size = 256
        shuffle_buffer_size = 20000
    else:
        batch_size = train_config['training'].get('batch_size', 256)
        shuffle_buffer_size = train_config['training'].get('shuffle_buffer_size', 20000)
    
    print(f"批处理大小: {batch_size}, Shuffle缓冲区大小: {shuffle_buffer_size}")
    
    # 设置训练集批处理和shuffle
    train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 设置验证集批处理
    validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 创建一个完整的数据集用于整体分析
    dataset = dataset_with_userid.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 保存用户划分信息到日志
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
    
    # 验证数据集并获取输入签名
    input_signature = validate_dataset(dataset)
    print("输入签名:", input_signature)
    
    # 检查数据集
    inspect_datasets(dataset, train_dataset, validation_dataset)
    
    # 加载特征配置，排除user_id特征
    exclude_features = ['user_id']
    print(f"\n=== 排除以下特征 ===\n{exclude_features}")
    pipelines_config = load_feature_config(exclude_features=exclude_features)
    
    # 创建并编译模型
    model = MLP(pipelines_config, train_config=train_config)
    
    # 打印所有特征管道的配置信息
    print("\n=== 特征管道配置信息 ===")
    if hasattr(model, 'feature_pipelines'):
        for i, (feature_name, processors) in enumerate(model.feature_pipelines):
            print(f"特征管道 #{i+1}: 特征名称 = {feature_name}")
            print(f"  处理器链: {[p.__class__.__name__ for p in processors]}")
    else:
        print("模型未定义特征管道")
    
    # 配置优化器
    if train_config is None:
        learning_rate = 0.0005
        weight_decay = 0.001
    else:
        learning_rate = train_config['training'].get('lr', 0.0005)
        weight_decay = train_config['training'].get('weight_decay', 0.001)
    
    # 使用带有学习率衰减的优化器
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=100,
        decay_rate=0.96,
        staircase=True)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    
    # 尝试对一个批次应用模型
    print("尝试对一个批次应用模型...")
    try:
        for batch in dataset.take(1):
            features, labels = batch
            print("特征键:", list(features.keys()))
            predictions = model(features)
            print("预测成功，结果形状:", predictions.shape)
    except Exception as e:
        print(f"模型预测失败: {e}")
        # 如果预测失败，暂时跳过训练
        print("跳过训练步骤")
        return
    
    # 训练模型
    history = train_model(model, dataset, train_dataset, validation_dataset, train_config=train_config)
    
    print(f"模型训练完成")
    print(f"训练集最终AUC: {history.history['auc'][-1]:.6f}")
    print(f"验证集最终AUC: {history.history['val_auc'][-1]:.6f}")
    
    # 评估特征重要性
    print("\n开始评估特征重要性...")
    feature_importance = check_feature_importance(model, validation_dataset, train_config=train_config)
    
    # 绘制特征重要性图
    plot_feature_importance(feature_importance)


if __name__ == "__main__":
    main() 