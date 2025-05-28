#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型工具模块，提供模型创建和测试功能
"""

import tensorflow as tf
from src.utils.config_loader import load_feature_config


def create_and_compile_model(model_class, train_config):
    """
    创建并编译模型
    
    参数:
        model_class: 模型类
        train_config: 训练配置字典
        
    返回:
        model: 编译后的模型
    """
    # 加载特征配置，排除user_id特征
    exclude_features = ['user_id']
    print(f"\n=== 排除以下特征 ===\n{exclude_features}")
    pipelines_config = load_feature_config(exclude_features=exclude_features)
    
    # 创建模型
    model = model_class(pipelines_config, train_config=train_config)
    
    # 打印特征管道配置信息
    print_feature_pipelines(model)
    
    # 配置优化器参数
    learning_rate = train_config['training'].get('lr', 0.0005) if train_config else 0.0005
    weight_decay = train_config['training'].get('weight_decay', 0.001) if train_config else 0.001
    
    # 使用带有学习率衰减的优化器
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=100,
        decay_rate=0.96,
        staircase=True)
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    
    return model


def print_feature_pipelines(model):
    """
    打印模型的特征管道配置信息
    
    参数:
        model: 模型实例
    """
    print("\n=== 特征管道配置信息 ===")
    if hasattr(model, 'feature_pipelines'):
        for i, (feature_name, processors) in enumerate(model.feature_pipelines):
            print(f"特征管道 #{i+1}: 特征名称 = {feature_name}")
            print(f"  处理器链: {[p.__class__.__name__ for p in processors]}")
    else:
        print("模型未定义特征管道")


def test_model_on_batch(model, dataset):
    """
    在单个批次上测试模型
    
    参数:
        model: 模型实例
        dataset: 测试数据集
        
    返回:
        成功: True，失败: False
    """
    print("尝试对一个批次应用模型...")
    try:
        for batch in dataset.take(1):
            features, labels = batch
            print("特征键:", list(features.keys()))
            predictions = model(features)
            print("预测成功，结果形状:", predictions.shape)
        return True
    except Exception as e:
        print(f"模型预测失败: {e}")
        return False 