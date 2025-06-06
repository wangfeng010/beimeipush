#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型工具模块，提供模型创建和测试功能
"""

from typing import Type, Dict, List, Any, Optional, Union, Tuple

import tensorflow as tf
from src.utils.config_utils import load_feature_config


def create_and_compile_model(
    model_class: Type[tf.keras.Model], 
    train_config: Optional[Dict[str, Any]] = None
) -> tf.keras.Model:
    """
    创建并编译模型
    
    Args:
        model_class: 模型类
        train_config: 训练配置字典
        
    Returns:
        编译后的模型实例
    """
    # 加载特征配置
    pipelines_config = _load_feature_pipelines_config()
    
    # 创建模型
    model = model_class(pipelines_config, train_config=train_config)
    
    # 打印特征管道配置信息
    print_feature_pipelines(model)
    
    # 配置和编译模型
    _compile_model(model, train_config)
    
    return model


def _load_feature_pipelines_config() -> List[Dict[str, Any]]:
    """
    加载特征管道配置
    
    Returns:
        特征管道配置列表
    """
    print(f"\n=== 从配置文件加载特征设置 ===")
    
    # 从配置文件加载配置（使用当前配置，包含完整排除逻辑）
    pipelines_config = load_feature_config()
    
    return pipelines_config


def _compile_model(
    model: tf.keras.Model, 
    train_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    配置并编译模型
    
    Args:
        model: 待编译的模型
        train_config: 训练配置字典
    """
    # 从配置获取优化器参数
    if train_config and 'training' in train_config:
        learning_rate = train_config['training'].get('lr', 0.0005)
        weight_decay = train_config['training'].get('weight_decay', 0.001)
    else:
        learning_rate = 0.0005
        weight_decay = 0.001
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )


def _get_config_value(
    config: Optional[Dict[str, Any]], 
    path: str, 
    default_value: Any
) -> Any:
    """
    从配置字典中获取嵌套值
    
    Args:
        config: 配置字典
        path: 点分隔的路径，例如 'training.lr'
        default_value: 未找到值时的默认值
    
    Returns:
        配置值或默认值
    """
    if config is None:
        return default_value
    
    keys = path.split('.')
    current = config
    
    try:
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default_value
        return current
    except Exception:
        return default_value


def print_feature_pipelines(model: tf.keras.Model) -> None:
    """
    打印模型的特征管道配置信息
    
    Args:
        model: 模型实例
    """
    print("\n=== 特征管道配置信息 ===")
    if hasattr(model, 'feature_pipelines'):
        for i, (feature_name, processors) in enumerate(model.feature_pipelines):
            print(f"特征管道 #{i+1}: 特征名称 = {feature_name}")
            print(f"  处理器链: {[p.__class__.__name__ for p in processors]}")
    else:
        print("模型未定义特征管道")


def test_model_on_batch(
    model: tf.keras.Model, 
    dataset: tf.data.Dataset
) -> bool:
    """
    在单个批次上测试模型
    
    Args:
        model: 模型实例
        dataset: 测试数据集
        
    Returns:
        True表示测试成功，False表示测试失败
    """
    print("尝试对一个批次应用模型...")
    try:
        for batch in dataset.take(1):
            features, labels = batch
            _test_model_prediction(model, features)
        return True
    except Exception as e:
        print(f"模型预测失败: {e}")
        return False


def _test_model_prediction(
    model: tf.keras.Model, 
    features: Dict[str, tf.Tensor]
) -> None:
    """
    使用给定特征测试模型预测
    
    Args:
        model: 模型实例
        features: 特征字典
    """
    print("特征键:", list(features.keys()))
    predictions = model(features, training=False)
    print("预测成功，结果形状:", predictions.shape) 