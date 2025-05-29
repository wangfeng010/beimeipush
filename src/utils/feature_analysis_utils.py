#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature Analysis Utilities - 特征分析工具
"""

import os
import json
import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def check_feature_importance(
    model: tf.keras.Model, 
    dataset: tf.data.Dataset, 
    train_config: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    使用排列重要性方法检查特征重要性
    
    Args:
        model: 训练好的模型
        dataset: 测试数据集
        train_config: 训练配置
    
    Returns:
        特征重要性字典，其中键为特征名，值为重要性分数(基线AUC - 随机化后的AUC)
    """
    # 设置参数
    num_batches = train_config.get('eval_batches', 5) if train_config else 5
    num_repeats = train_config.get('importance_repeats', 3) if train_config else 3
    
    # 使用采样数据进行评估
    print(f"使用 {num_batches} 批次数据评估特征重要性...")
    sample_dataset = dataset.take(num_batches)
    
    # 获取基线性能
    baseline_auc, all_labels, all_preds = _calculate_baseline_performance(model, sample_dataset, num_batches)
    print(f"基线 AUC: {baseline_auc:.4f}")
    
    # 获取所有特征名称
    feature_names = _get_feature_names(sample_dataset)
    
    # 评估每个特征的重要性
    feature_importance = _evaluate_all_features(
        model, sample_dataset, feature_names, 
        baseline_auc, num_batches, num_repeats
    )
    
    # 处理和保存结果
    sorted_importance = _process_and_save_results(feature_importance)
    
    return sorted_importance


def _calculate_baseline_performance(
    model: tf.keras.Model, 
    dataset: tf.data.Dataset, 
    num_batches: int
) -> Tuple[float, List[float], List[float]]:
    """
    计算模型在原始数据集上的基线性能
    
    Args:
        model: 训练好的模型
        dataset: 评估数据集
        num_batches: 要处理的批次数
    
    Returns:
        baseline_auc: 基线AUC分数
        all_labels: 所有真实标签
        all_preds: 所有预测值
    """
    all_labels = []
    all_preds = []
    
    # 收集预测和实际标签
    print("收集基线预测...")
    for i, (x, y) in enumerate(dataset):
        try:
            # 使用模型进行预测
            preds = model(x, training=False)
            # 确保预测和标签是1D数组
            y_pred = preds.numpy().flatten()
            y_true = y.numpy().flatten()
            
            all_preds.extend(y_pred)
            all_labels.extend(y_true)
            print(f"已处理批次 {i+1}/{num_batches}")
        except Exception as e:
            print(f"处理批次 {i+1} 时出错: {str(e)}")
    
    # 计算基线AUC
    baseline_auc = roc_auc_score(all_labels, all_preds)
    return baseline_auc, all_labels, all_preds


def _get_feature_names(dataset: tf.data.Dataset) -> List[str]:
    """
    从数据集中获取特征名称
    
    Args:
        dataset: 数据集
    
    Returns:
        特征名称列表
    """
    first_batch = next(iter(dataset))
    return list(first_batch[0].keys())


def _evaluate_all_features(
    model: tf.keras.Model, 
    dataset: tf.data.Dataset, 
    feature_names: List[str], 
    baseline_auc: float,
    num_batches: int,
    num_repeats: int
) -> Dict[str, float]:
    """
    评估所有特征的重要性
    
    Args:
        model: 训练好的模型
        dataset: 评估数据集
        feature_names: 特征名称列表
        baseline_auc: 基线AUC分数
        num_batches: 要处理的批次数
        num_repeats: 每个特征的重复评估次数
    
    Returns:
        特征重要性字典
    """
    feature_importance = {}
    
    for feature_name in feature_names:
        importance = _evaluate_single_feature(
            model, dataset, feature_name, baseline_auc, num_repeats
        )
        feature_importance[feature_name] = importance
        print(f"  特征 {feature_name} 重要性: {importance:.6f}")
    
    return feature_importance


def _evaluate_single_feature(
    model: tf.keras.Model, 
    dataset: tf.data.Dataset, 
    feature_name: str, 
    baseline_auc: float,
    num_repeats: int
) -> float:
    """
    评估单个特征的重要性，通过比较基线AUC和特征随机化后的AUC计算
    
    Args:
        model: 训练好的模型
        dataset: 评估数据集
        feature_name: 要评估的特征名称
        baseline_auc: 基线AUC分数
        num_repeats: 重复评估次数
    
    Returns:
        特征重要性分数，计算方式为基线AUC减去随机化后的AUC
    """
    print(f"评估特征 '{feature_name}' 的重要性...")
    feature_aucs = []
    
    # 执行多次重复以增加稳定性
    for repeat in range(num_repeats):
        permuted_auc = _single_feature_permutation_test(model, dataset, feature_name)
        if permuted_auc is not None:
            feature_aucs.append(permuted_auc)
    
    # 计算平均AUC
    if feature_aucs:
        avg_permuted_auc = np.mean(feature_aucs)
        importance = baseline_auc - avg_permuted_auc
        return importance
    else:
        print(f"  无法评估特征 {feature_name} 的重要性")
        return 0.0


def _single_feature_permutation_test(
    model: tf.keras.Model, 
    dataset: tf.data.Dataset, 
    feature_name: str
) -> Optional[float]:
    """
    对单个特征进行排列测试
    
    Args:
        model: 训练好的模型
        dataset: 评估数据集
        feature_name: 要评估的特征名称
    
    Returns:
        特征随机化后的AUC分数，如果测试失败则返回None
    """
    all_preds_permuted = []
    all_labels_permuted = []
    
    # 遍历数据集
    for i, (x, y) in enumerate(dataset):
        try:
            # 创建x的副本
            x_copy = {k: tf.identity(v) for k, v in x.items()}
            
            # 随机化特定特征
            if x_copy[feature_name].shape[0] > 1:  # 确保批次有多个样本
                x_copy[feature_name] = tf.random.shuffle(x_copy[feature_name])
            
            # 进行预测
            preds = model(x_copy, training=False)
            all_preds_permuted.extend(preds.numpy().flatten())
            all_labels_permuted.extend(y.numpy().flatten())
        except Exception as e:
            print(f"  预测错误: {str(e)}")
            continue
    
    # 如果收集了足够的预测
    if len(all_preds_permuted) > 0:
        # 计算特征随机化后的AUC
        return roc_auc_score(all_labels_permuted, all_preds_permuted)
    
    return None


def _process_and_save_results(feature_importance: Dict[str, float]) -> Dict[str, float]:
    """
    处理和保存特征重要性结果
    
    Args:
        feature_importance: 特征重要性字典
    
    Returns:
        按重要性排序的特征字典
    """
    # 按重要性排序
    sorted_importance = {
        k: v for k, v in sorted(
            feature_importance.items(), 
            key=lambda item: abs(item[1]), 
            reverse=True
        )
    }
    
    # 打印特征重要性排名
    print("\n特征重要性排名:")
    for i, (feature, importance) in enumerate(sorted_importance.items()):
        print(f"{i+1}. {feature}: {importance:.6f}")
    
    # 确保日志目录存在
    os.makedirs("./logs", exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存特征重要性到文件
    importance_file = f"./logs/feature_importance_{timestamp}.json"
    with open(importance_file, 'w') as f:
        json.dump(sorted_importance, f, indent=2)
    
    print(f"特征重要性已保存到: {importance_file}")
    
    return sorted_importance


def plot_feature_importance(
    feature_importance: Dict[str, float],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    绘制特征重要性图表
    
    Args:
        feature_importance: 特征重要性字典，键为特征名，值为重要性分数
        figsize: 图表尺寸，默认为(10, 6)
        save_path: 保存路径，如果为None则自动生成
    """
    # 确保日志目录存在
    os.makedirs("./logs", exist_ok=True)
    
    # 按重要性排序
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    # 提取特征名称和重要性分数
    features = [x[0] for x in sorted_features]
    importance = [x[1] for x in sorted_features]
    
    # 限制显示的特征数量以提高可读性
    if len(features) > 15:
        features = features[:15]
        importance = importance[:15]
        print("注意: 只显示前15个最重要的特征")
    
    # 创建图表
    plt.figure(figsize=figsize)
    bars = plt.barh(features, importance, color='skyblue')
    
    # 为负值添加不同的颜色
    for i, v in enumerate(importance):
        if v < 0:
            bars[i].set_color('salmon')
    
    # 添加标签和标题
    plt.xlabel('重要性分数 (基线AUC - 随机化后的AUC)')
    plt.ylabel('特征')
    plt.title('特征重要性排名')
    plt.tight_layout()
    
    # 保存图表
    if save_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'./logs/feature_importance_{timestamp}.png'
    
    plt.savefig(save_path, dpi=300)
    print(f"特征重要性图表已保存到: {save_path}")
    plt.close() 