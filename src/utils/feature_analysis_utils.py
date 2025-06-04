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

# 添加配置工具导入
from .config_utils import load_feature_config


def check_feature_importance(
    model: tf.keras.Model, 
    dataset: tf.data.Dataset, 
    train_config: Optional[Dict[str, Any]] = None,
    feat_config_path: str = "config/feat.yml"
) -> Dict[str, float]:
    """
    使用排列重要性方法检查特征重要性
    
    Args:
        model: 训练好的模型
        dataset: 测试数据集
        train_config: 训练配置
        feat_config_path: 特征配置文件路径
    
    Returns:
        特征重要性字典，其中键为特征名，值为重要性分数(基线AUC - 随机化后的AUC)
    """
    # 设置参数
    num_batches = train_config.get('eval_batches', 5) if train_config else 5
    num_repeats = train_config.get('importance_repeats', 3) if train_config else 3
    
    # 使用采样数据进行评估
    print(f"使用 {num_batches} 批次数据评估特征重要性...")
    print(f"特征配置文件: {feat_config_path}")
    sample_dataset = dataset.take(num_batches)
    
    # 获取基线性能
    baseline_auc, all_labels, all_preds = _calculate_baseline_performance(model, sample_dataset)
    print(f"基线 AUC: {baseline_auc:.4f}")
    
    # 获取特征名称
    input_feature_names = _get_feature_names(sample_dataset)
    
    # 🚀 直接从配置文件获取处理后的特征名称
    processed_feature_names = _get_processed_feature_names_from_config(feat_config_path)
    
    print(f"原始输入特征数量: {len(input_feature_names)}")
    print(f"处理后特征数量: {len(processed_feature_names)}")
    
    # 评估原始输入特征的重要性
    print("\n评估原始输入特征重要性:")
    input_feature_importance = _evaluate_features(
        model, sample_dataset, input_feature_names, 
        baseline_auc, num_repeats, is_processed=False
    )
    
    # 评估处理后特征的重要性
    feature_importance = input_feature_importance
    if processed_feature_names:
        print("\n评估处理后特征重要性:")
        processed_feature_importance = _evaluate_processed_features(
            model, sample_dataset, processed_feature_names,
            baseline_auc, num_repeats, feat_config_path
        )
        # 合并特征重要性结果
        feature_importance.update(processed_feature_importance)
    
    # 处理和保存结果
    sorted_importance = _process_and_save_results(feature_importance)
    
    return sorted_importance


def _calculate_baseline_performance(
    model: tf.keras.Model, 
    dataset: tf.data.Dataset
) -> Tuple[float, List[float], List[float]]:
    """计算模型在原始数据集上的基线性能"""
    all_labels = []
    all_preds = []
    
    for i, (x, y) in enumerate(dataset):
        try:
            preds = model(x, training=False)
            y_pred = preds.numpy().flatten()
            y_true = y.numpy().flatten()
            
            all_preds.extend(y_pred)
            all_labels.extend(y_true)
            print(f"已处理批次 {i+1}")
        except Exception as e:
            print(f"处理批次时出错: {str(e)}")
    
    baseline_auc = roc_auc_score(all_labels, all_preds)
    return baseline_auc, all_labels, all_preds


def _get_feature_names(dataset: tf.data.Dataset) -> List[str]:
    """从数据集中获取输入特征名称"""
    first_batch = next(iter(dataset))
    return list(first_batch[0].keys())


def _get_processed_feature_names_from_config(config_path: str = "config/feat.yml") -> List[str]:
    """
    从配置文件中获取处理后的特征名称
    
    Args:
        config_path: 特征配置文件路径
    
    Returns:
        处理后特征名称列表
    """
    try:
        # 解析配置映射
        mappings = _parse_feature_config_mappings(config_path)
        feature_names = list(mappings.keys())
        
        print(f"从配置文件解析到的处理后特征: {feature_names}")
        return feature_names
        
    except Exception as e:
        print(f"从配置文件获取特征名称失败: {str(e)}")
        return []


def _evaluate_features(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    feature_names: List[str],
    baseline_auc: float,
    num_repeats: int,
    is_processed: bool = False
) -> Dict[str, float]:
    """评估特征重要性"""
    feature_importance = {}
    
    for feature_name in feature_names:
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
    num_repeats: int,
    feat_config_path: str
) -> Dict[str, float]:
    """评估处理后特征的重要性"""
    feature_importance = {}
    
    # 🚀 新方案：从配置文件自动解析特征映射
    print("正在从配置文件解析特征映射关系...")
    config_mappings = _parse_feature_config_mappings(feat_config_path)
    
    # 从模型中获取特征映射（作为补充）
    model_mappings = {}
    if hasattr(model, 'feature_pipelines'):
        for _, processors in model.feature_pipelines:
            if processors and len(processors) > 0:
                last_processor = processors[-1]
                first_processor = processors[0]
                
                if hasattr(last_processor, 'col_out') and last_processor.col_out:
                    output_name = last_processor.col_out
                    
                    if hasattr(first_processor, 'col_in'):
                        input_features = [first_processor.col_in] if isinstance(first_processor.col_in, str) else first_processor.col_in
                        model_mappings[output_name] = input_features
    
    # 合并映射：优先使用配置文件解析的映射，然后是模型映射
    feature_mapping = {}
    feature_mapping.update(model_mappings)  # 先添加模型映射
    feature_mapping.update(config_mappings)  # 配置映射覆盖模型映射
    
    print(f"总共解析到 {len(feature_mapping)} 个特征映射关系")
    
    # 评估特征重要性
    for processed_name in feature_names:
        # 查找对应输入特征
        input_features = feature_mapping.get(processed_name, [])
        
        # 回退策略：如果没有找到映射，尝试简单的名称匹配
        if not input_features and "_emb" in processed_name:
            possible_input = processed_name.replace("_emb", "")
            dataset_feature_names = _get_feature_names(dataset)
            if possible_input in dataset_feature_names:
                input_features = [possible_input]
                print(f"  使用回退策略为 {processed_name} 找到输入: {possible_input}")
        
        if input_features:
            print(f"评估处理后特征 '{processed_name}' (输入: {', '.join(input_features)})的重要性...")
            
            # 对于BERT特征，使用特殊的评估方法
            if _is_bert_feature(processed_name):
                importance = _evaluate_bert_feature_importance(
                    model, dataset, processed_name, input_features, baseline_auc, num_repeats
                )
            else:
                # 计算重要性
                importances = [_evaluate_single_feature(model, dataset, input_name, baseline_auc, num_repeats) 
                              for input_name in input_features]
                # 取最大值作为最终重要性
                importance = max(importances) if importances else 0.0
            
            feature_importance[processed_name] = importance
            print(f"  处理后特征 {processed_name} 重要性: {importance:.6f}")
        else:
            print(f"  ⚠️  警告: 找不到处理后特征 {processed_name} 的输入特征")
            feature_importance[processed_name] = 0.0
    
    return feature_importance


def _is_bert_feature(feature_name: str) -> bool:
    """检查是否为BERT特征"""
    bert_indicators = ['precomputed', 'bert', 'transformer']
    return any(indicator in feature_name.lower() for indicator in bert_indicators)


def _evaluate_single_feature(
    model: tf.keras.Model, 
    dataset: tf.data.Dataset, 
    feature_name: str, 
    baseline_auc: float,
    num_repeats: int
) -> float:
    """评估单个特征的重要性"""
    print(f"评估特征 '{feature_name}' 的重要性...")
    feature_aucs = []
    
    for _ in range(num_repeats):
        all_preds = []
        all_labels = []
        
        for x, y in dataset:
            try:
                # 创建特征副本并随机化
                x_copy = {k: tf.identity(v) for k, v in x.items()}
                if x_copy[feature_name].shape[0] > 1:
                    x_copy[feature_name] = tf.random.shuffle(x_copy[feature_name])
                
                # 预测
                preds = model(x_copy, training=False)
                all_preds.extend(preds.numpy().flatten())
                all_labels.extend(y.numpy().flatten())
            except Exception as e:
                print(f"  预测错误: {str(e)}")
                continue
        
        # 计算AUC
        if all_preds:
            permuted_auc = roc_auc_score(all_labels, all_preds)
            feature_aucs.append(permuted_auc)
    
    # 计算重要性
    if feature_aucs:
        avg_permuted_auc = np.mean(feature_aucs)
        return baseline_auc - avg_permuted_auc
    else:
        print(f"  无法评估特征 {feature_name} 的重要性")
        return 0.0


def _evaluate_bert_feature_importance(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    feature_name: str,
    input_features: List[str],
    baseline_auc: float,
    num_repeats: int
) -> float:
    """专门评估BERT特征的重要性"""
    print(f"  使用BERT特征专用方法评估 '{feature_name}'...")
    
    # 对于BERT特征，我们直接评估相关的文本输入特征
    # 因为BERT特征本质上是对文本内容的表示
    feature_aucs = []
    
    for _ in range(num_repeats):
        all_preds = []
        all_labels = []
        
        for x, y in dataset:
            try:
                # 创建特征副本并随机化相关的文本特征
                x_copy = {k: tf.identity(v) for k, v in x.items()}
                
                # 随机化所有相关的输入特征
                for input_feature in input_features:
                    if input_feature in x_copy and x_copy[input_feature].shape[0] > 1:
                        x_copy[input_feature] = tf.random.shuffle(x_copy[input_feature])
                
                # 预测
                preds = model(x_copy, training=False)
                all_preds.extend(preds.numpy().flatten())
                all_labels.extend(y.numpy().flatten())
            except Exception as e:
                print(f"    BERT特征评估错误: {str(e)}")
                continue
        
        # 计算AUC
        if all_preds:
            permuted_auc = roc_auc_score(all_labels, all_preds)
            feature_aucs.append(permuted_auc)
    
    # 计算重要性
    if feature_aucs:
        avg_permuted_auc = np.mean(feature_aucs)
        return baseline_auc - avg_permuted_auc
    else:
        print(f"    无法评估BERT特征 {feature_name} 的重要性")
        return 0.0


def _process_and_save_results(feature_importance: Dict[str, float]) -> Dict[str, float]:
    """处理和保存特征重要性结果"""
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
    
    # 保存结果
    os.makedirs("./logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
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
    """绘制特征重要性图表"""
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
        # 是否为BERT特征
        is_bert = any(bert_class in name for bert_class in ['BertEmbedding', 'PrecomputedEmbedding'])
        feature_display = f"[BERT] {name}" if is_bert else name
        colors.append('darkred' if is_bert and imp > 0 else 
                     'salmon' if is_bert else 
                     'teal' if imp > 0 else 'skyblue')
        
        features.append(feature_display)
        importance.append(imp)
    
    # 限制显示的特征数量
    max_features = 20
    if len(features) > max_features:
        features = features[:max_features]
        importance = importance[:max_features]
        colors = colors[:max_features]
        print(f"注意: 只显示前{max_features}个最重要的特征")
    
    # 创建图表
    plt.figure(figsize=figsize)
    plt.barh(features, importance, color=colors)
    
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


def _parse_feature_config_mappings(config_path: str = "config/feat.yml") -> Dict[str, List[str]]:
    """
    从feat.yml配置文件自动解析特征映射关系
    
    Args:
        config_path: 特征配置文件路径
    
    Returns:
        特征映射字典，格式为 {输出特征名: [输入特征列表]}
    """
    print(f"正在解析特征配置文件: {config_path}")
    
    try:
        # 加载特征配置
        feature_configs = load_feature_config(config_path, exclude_features=[])  # 暂时不排除任何特征
        
        feature_mappings = {}
        
        for config in feature_configs:
            feat_name = config.get('feat_name')
            operations = config.get('operations', [])
            
            if not feat_name or not operations:
                continue
            
            # 获取输入特征
            input_features = _extract_input_features_from_operations(operations)
            
            # 获取输出特征名
            output_feature = _extract_output_feature_from_operations(operations)
            
            if output_feature and input_features:
                # 处理特殊的BERT特征
                actual_inputs = _resolve_bert_feature_inputs(feat_name, input_features, operations)
                feature_mappings[output_feature] = actual_inputs
                
                print(f"  解析特征: {output_feature} <- {actual_inputs}")
            elif output_feature:
                # 即使没有找到输入特征，也要记录这个输出特征
                # 这样可以确保所有特征都被评估
                if input_features:
                    feature_mappings[output_feature] = input_features
                else:
                    # 尝试从feat_name推断原始输入
                    inferred_input = _infer_input_from_feature_name(feat_name)
                    feature_mappings[output_feature] = [inferred_input] if inferred_input else [feat_name]
                print(f"  解析特征 (回退): {output_feature} <- {feature_mappings[output_feature]}")
        
        print(f"成功解析了 {len(feature_mappings)} 个特征映射关系")
        return feature_mappings
        
    except Exception as e:
        print(f"解析特征配置失败: {str(e)}")
        print("回退到基本映射策略...")
        return {}


def _extract_input_features_from_operations(operations: List[Dict[str, Any]]) -> List[str]:
    """从操作序列中提取**原始输入特征**，过滤中间处理步骤"""
    if not operations:
        return []
    
    # 收集所有输入特征
    all_inputs = []
    for operation in operations:
        # 检查单个输入
        if 'col_in' in operation:
            col_in = operation['col_in']
            if isinstance(col_in, str):
                all_inputs.append(col_in)
        
        # 检查多个输入
        if 'col_in_list' in operation:
            col_in_list = operation['col_in_list']
            if isinstance(col_in_list, list):
                for col in col_in_list:
                    if isinstance(col, str):
                        all_inputs.append(col)
    
    # 收集所有输出特征（这些是中间步骤产生的）
    all_outputs = []
    for operation in operations:
        if 'col_out' in operation:
            col_out = operation['col_out']
            if isinstance(col_out, str):
                all_outputs.append(col_out)
    
    # 🎯 关键逻辑：只保留那些**不是由前面步骤产生的**输入特征
    # 即：不在outputs列表中的inputs才是真正的原始输入
    original_inputs = []
    for input_feature in all_inputs:
        if input_feature not in all_outputs:
            if input_feature not in original_inputs:  # 去重
                original_inputs.append(input_feature)
    
    return original_inputs


def _extract_output_feature_from_operations(operations: List[Dict[str, Any]]) -> Optional[str]:
    """从操作序列中提取最终输出特征名"""
    if not operations:
        return None
    
    # 取最后一个操作的输出
    last_operation = operations[-1]
    return last_operation.get('col_out')


def _resolve_bert_feature_inputs(
    feat_name: str, 
    input_features: List[str], 
    operations: List[Dict[str, Any]]
) -> List[str]:
    """
    解析BERT特征的实际输入
    
    BERT特征的特殊之处：
    - 配置中输入是create_time（用于索引）
    - 实际处理的是推送内容
    """
    # 检查是否为BERT特征
    is_bert_feature = any(
        op.get('func_name') == 'PrecomputedEmbedding' 
        for op in operations
    )
    
    if not is_bert_feature:
        return input_features
    
    # 根据特征名称推断实际处理的内容
    bert_mappings = {
        'title_content_precomputed_emb': ['push_title', 'push_content'],
        'push_title_bert_emb': ['push_title'],
        'push_content_bert_emb': ['push_content']
    }
    
    # 检查是否匹配已知的BERT特征
    for pattern, actual_inputs in bert_mappings.items():
        if pattern in feat_name:
            print(f"    检测到BERT特征 {feat_name}，映射到实际输入: {actual_inputs}")
            return actual_inputs
    
    # 如果是未知的BERT特征，尝试从特征名推断
    if 'bert' in feat_name.lower() or 'precomputed' in feat_name.lower():
        if 'title' in feat_name and 'content' in feat_name:
            return ['push_title', 'push_content']
        elif 'title' in feat_name:
            return ['push_title']
        elif 'content' in feat_name:
            return ['push_content']
    
    # 默认返回原始输入特征
    return input_features 


def _infer_input_from_feature_name(feat_name: str) -> Optional[str]:
    """
    从特征名称推断原始输入特征
    
    Args:
        feat_name: 特征名称，如 "watchlists_emb"
    
    Returns:
        推断的输入特征名称，如 "watchlists"
    """
    # 移除常见的后缀
    suffixes_to_remove = ['_emb', '_embedding', '_feature']
    
    inferred = feat_name
    for suffix in suffixes_to_remove:
        if inferred.endswith(suffix):
            inferred = inferred[:-len(suffix)]
            break
    
    # 如果推断结果与原名称不同，则返回推断结果
    if inferred != feat_name:
        return inferred
    
    return None 