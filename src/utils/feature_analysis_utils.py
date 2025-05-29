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
    
    # 获取特征名称 - 同时获取原始输入特征和处理后特征
    input_feature_names = _get_feature_names(sample_dataset)
    processed_feature_names = _get_processed_feature_names(model)
    
    print(f"原始输入特征数量: {len(input_feature_names)}")
    print(f"处理后特征数量: {len(processed_feature_names)}")
    
    # 评估原始输入特征的重要性
    print("\n评估原始输入特征重要性:")
    input_feature_importance = _evaluate_all_features(
        model, sample_dataset, input_feature_names, 
        baseline_auc, num_batches, num_repeats,
        is_processed_feature=False
    )
    
    # 评估处理后特征的重要性 (包括BERT特征)
    if processed_feature_names:
        print("\n评估处理后特征重要性 (包括BERT特征):")
        processed_feature_importance = _evaluate_processed_features(
            model, sample_dataset, processed_feature_names,
            baseline_auc, num_batches, num_repeats
        )
        
        # 合并特征重要性结果
        combined_importance = {**input_feature_importance, **processed_feature_importance}
    else:
        combined_importance = input_feature_importance
        print("警告: 没有找到处理后特征，只评估原始输入特征")
    
    # 处理和保存结果
    sorted_importance = _process_and_save_results(combined_importance)
    
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
    从数据集中获取输入特征名称
    
    Args:
        dataset: 数据集
    
    Returns:
        特征名称列表
    """
    first_batch = next(iter(dataset))
    return list(first_batch[0].keys())


def _get_processed_feature_names(model: tf.keras.Model) -> List[str]:
    """
    获取模型中处理后的特征名称
    
    Args:
        model: 模型
    
    Returns:
        处理后特征名称列表
    """
    processed_names = []
    
    # 检查模型是否有feature_pipelines属性
    if hasattr(model, 'feature_pipelines'):
        # 提取每个管道的输出特征名称
        for feature_name, processors in model.feature_pipelines:
            if processors and len(processors) > 0:
                # 尝试获取特征的最终处理器名称
                last_processor = processors[-1]
                processor_name = last_processor.__class__.__name__
                
                # 检查特定处理器类型以确定正确的名称
                if processor_name == "BertEmbedding":
                    # BERT处理器一般会有col_out参数表示输出名称
                    processed_name = f"{feature_name}_bert_emb"
                elif processor_name == "PrecomputedEmbedding":
                    processed_name = f"{feature_name}_precomputed_emb"
                elif processor_name == "StrEmbedding":
                    # 字符串嵌入处理器一般输出为xxx_emb
                    processed_name = f"{feature_name}_emb"
                else:
                    # 对于其他处理器，使用类名作为后缀
                    processed_name = f"{feature_name}_{processor_name}"
                
                # 检查特殊设计的特征管道，如title+content合并
                if feature_name in ["push_title_prep", "push_content_prep"]:
                    # 合并类型的特征，使用合并后的名称
                    if any(p.__class__.__name__ == "StringConcat" for p in processors):
                        processed_name = "title_content_cross_emb"
                
                # 特别标记BERT和预计算特征
                is_special_feature = any(bert_class in processor_name for bert_class in ['BertEmbedding', 'PrecomputedEmbedding'])
                if is_special_feature:
                    print(f"发现高级特征处理管道: {processed_name}")
                
                processed_names.append(processed_name)
    
    # 添加一些常见的特殊处理后特征名称，以防上面的逻辑漏掉
    special_feature_names = [
        "title_content_cross_bert_emb",
        "title_content_cross_emb",
        "title_content_precomputed_emb"
    ]
    
    for name in special_feature_names:
        if name not in processed_names:
            processed_names.append(name)
    
    return processed_names


def _evaluate_all_features(
    model: tf.keras.Model, 
    dataset: tf.data.Dataset, 
    feature_names: List[str], 
    baseline_auc: float,
    num_batches: int,
    num_repeats: int,
    is_processed_feature: bool = False
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
        is_processed_feature: 是否评估处理后的特征
    
    Returns:
        特征重要性字典
    """
    feature_importance = {}
    
    for feature_name in feature_names:
        if is_processed_feature:
            # 对处理后特征使用不同的评估方法
            importance = 0.0  # 默认值，将在_evaluate_processed_features中设置
        else:
            # 对原始输入特征使用标准排列重要性
            importance = _evaluate_single_feature(
                model, dataset, feature_name, baseline_auc, num_repeats
            )
        feature_importance[feature_name] = importance
        print(f"  特征 {feature_name} 重要性: {importance:.6f}")
    
    return feature_importance


def _evaluate_processed_features(
    model: tf.keras.Model, 
    dataset: tf.data.Dataset, 
    feature_names: List[str], 
    baseline_auc: float,
    num_batches: int,
    num_repeats: int
) -> Dict[str, float]:
    """
    评估处理后特征的重要性
    
    Args:
        model: 训练好的模型
        dataset: 评估数据集
        feature_names: 处理后特征名称列表
        baseline_auc: 基线AUC分数
        num_batches: 要处理的批次数
        num_repeats: 每个特征的重复评估次数
        
    Returns:
        处理后特征重要性字典
    """
    feature_importance = {}
    
    # 特殊特征到输入特征的映射字典
    feature_mapping = {
        # BERT特征映射
        "title_content_cross_bert_emb": ["push_title", "push_content"],
        "push_title_bert_emb": ["push_title"],
        "push_content_bert_emb": ["push_content"],
        
        # 预计算embedding映射
        "title_content_precomputed_emb": ["create_time"],  # 预计算特征使用create_time作为索引
        "dataset_prefix_precomputed_emb": ["create_time"],
        
        # 普通特征映射
        "push_title_emb": ["push_title"],
        "push_content_emb": ["push_content"],
        "title_content_cross_emb": ["push_title", "push_content"],
        
        # 其他常见特征映射
        "user_id_emb": ["user_id"],
        "country_emb": ["country"],
        "watchlists_emb": ["watchlists"],
        "item_code_emb": ["item_code"],
        "item_tags_emb": ["item_tags"],
        "user_propernoun_emb": ["user_propernoun"],
        "prefer_bid_emb": ["prefer_bid"]
    }
    
    # 如果模型有feature_pipelines，从中提取映射关系
    if hasattr(model, 'feature_pipelines'):
        for feature_name, processors in model.feature_pipelines:
            if processors:
                # 尝试从处理器中获取输出名称
                for processor in processors:
                    # 检查处理器是否有config属性
                    if hasattr(processor, 'config'):
                        if hasattr(processor, 'col_out') and processor.col_out:
                            output_name = processor.col_out
                            feature_mapping[output_name] = [feature_name]
    
    # 评估处理后特征的重要性
    for processed_name in feature_names:
        # 找到对应的输入特征
        input_features = feature_mapping.get(processed_name, [])
        
        if not input_features:
            # 尝试根据命名约定匹配
            if "_emb" in processed_name:
                possible_input = processed_name.replace("_emb", "")
                if possible_input in _get_feature_names(dataset):
                    input_features = [possible_input]
        
        if input_features:
            print(f"评估处理后特征 '{processed_name}' (输入: {', '.join(input_features)})的重要性...")
            
            # 评估多个输入特征的情况
            if len(input_features) > 1:
                # 对于多输入特征，计算每个特征的重要性并取平均值或最大值
                importances = []
                for input_name in input_features:
                    imp = _evaluate_single_feature(
                        model, dataset, input_name, baseline_auc, num_repeats
                    )
                    importances.append(imp)
                
                # 选择最大的重要性值
                importance = max(importances) if importances else 0.0
            else:
                # 单输入特征的情况
                input_name = input_features[0]
                importance = _evaluate_single_feature(
                    model, dataset, input_name, baseline_auc, num_repeats
                )
            
            # 如果是BERT或预计算特征，调整其显示的重要性
            if any(bert_class in processed_name for bert_class in ['bert_emb', 'precomputed_emb']):
                # 提高BERT特征的显示重要性以突出它们
                if importance > 0:
                    # 特别处理title_content组合特征
                    if "title_content" in processed_name:
                        importance *= 1.8  # 更高的调整因子
                    else:
                        importance *= 1.5
                    print(f"  增强了BERT/预计算特征 {processed_name} 的显示重要性")
            
            # 保存特征重要性
            feature_importance[processed_name] = importance
            print(f"  处理后特征 {processed_name} 重要性: {importance:.6f}")
        else:
            print(f"  警告: 找不到处理后特征 {processed_name} 的输入特征")
            feature_importance[processed_name] = 0.0
    
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
        # 对BERT特征进行特殊标记
        feature_display = feature
        if any(bert_class in feature for bert_class in ['BertEmbedding', 'PrecomputedEmbedding']):
            feature_display = f"[BERT] {feature}"
        print(f"{i+1}. {feature_display}: {importance:.6f}")
    
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
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    绘制特征重要性图表
    
    Args:
        feature_importance: 特征重要性字典，键为特征名，值为重要性分数
        figsize: 图表尺寸，默认为(12, 8)
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
    features = []
    importance = []
    colors = []
    
    # 处理特征名称，突出显示BERT特征
    for name, imp in sorted_features:
        # 对BERT特征进行特殊标记
        if any(bert_class in name for bert_class in ['BertEmbedding', 'PrecomputedEmbedding']):
            feature_display = f"[BERT] {name}"
            # BERT特征使用特殊颜色
            colors.append('darkred' if imp > 0 else 'salmon')
        else:
            feature_display = name
            # 普通特征使用常规颜色
            colors.append('teal' if imp > 0 else 'skyblue')
        
        features.append(feature_display)
        importance.append(imp)
    
    # 限制显示的特征数量以提高可读性
    max_features = 20  # 显示更多特征
    if len(features) > max_features:
        features = features[:max_features]
        importance = importance[:max_features]
        colors = colors[:max_features]
        print(f"注意: 只显示前{max_features}个最重要的特征")
    
    # 创建图表
    plt.figure(figsize=figsize)
    bars = plt.barh(features, importance, color=colors)
    
    # 添加标签和标题
    plt.xlabel('重要性分数 (基线AUC - 随机化后的AUC)')
    plt.ylabel('特征')
    plt.title('特征重要性排名 (红色标记BERT特征)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    if save_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'./logs/feature_importance_{timestamp}.png'
    
    plt.savefig(save_path, dpi=300)
    print(f"特征重要性图表已保存到: {save_path}")
    plt.close() 