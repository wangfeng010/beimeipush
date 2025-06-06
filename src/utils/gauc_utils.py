#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAUC (Group AUC) 计算工具 - 简化版
"""

import os
import json
import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score


def calculate_gauc_with_original_data(
    model: tf.keras.Model,
    processed_dataset: tf.data.Dataset,
    original_dataset: tf.data.Dataset,
    min_samples_per_user: int = 2,
    max_batches: Optional[int] = None,
    verbose: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """
    使用原始数据集中的user_id计算GAUC
    """
    if verbose:
        print("🔍 开始计算GAUC...")
    
    user_data = defaultdict(lambda: {'predictions': [], 'labels': []})
    total_samples = 0
    
    processed_iter = iter(processed_dataset)
    original_iter = iter(original_dataset)
    
    for batch_idx in range(max_batches if max_batches else 1000):
        # 获取数据批次
        processed_batch = next(processed_iter, None)
        original_batch = next(original_iter, None)
        
        if processed_batch is None or original_batch is None:
            break
            
        processed_features, processed_labels = processed_batch
        original_features, original_labels = original_batch
        
        # 检查user_id
        if "user_id" not in original_features:
            raise ValueError("原始数据集中缺少'user_id'字段")
        
        # 模型预测
        predictions = model(processed_features, training=False)
        
        # 提取数据
        user_ids = original_features["user_id"].numpy()
        pred_values = predictions.numpy().flatten()
        label_values = processed_labels.numpy().flatten()
        
        # 检查数组长度是否一致
        n_users = len(user_ids)
        n_preds = len(pred_values)
        n_labels = len(label_values)
        
        if n_users != n_preds or n_users != n_labels:
            if verbose:
                print(f"⚠️ 批次 {batch_idx} 数组长度不匹配: users={n_users}, preds={n_preds}, labels={n_labels}")
            # 取最小长度确保索引安全
            min_length = min(n_users, n_preds, n_labels)
            user_ids = user_ids[:min_length]
            pred_values = pred_values[:min_length]
            label_values = label_values[:min_length]
        
        # 按用户分组
        for i in range(len(user_ids)):
            user_id = user_ids[i]
            user_id_str = user_id.decode('utf-8') if isinstance(user_id, bytes) else str(user_id)
            user_data[user_id_str]['predictions'].append(float(pred_values[i]))
            user_data[user_id_str]['labels'].append(int(label_values[i]))
        
        total_samples += len(user_ids)
    
    return _calculate_gauc_from_user_data(user_data, min_samples_per_user, total_samples, verbose)


def _calculate_gauc_from_user_data(
    user_data: Dict[str, Dict[str, List]],
    min_samples_per_user: int,
    total_samples: int,
    verbose: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """从用户数据计算GAUC"""
    
    # 过滤有效用户
    valid_users = {}
    for user_id, data in user_data.items():
        if len(data['predictions']) >= min_samples_per_user:
            unique_labels = set(data['labels'])
            if len(unique_labels) > 1:  # 需要正负样本
                valid_users[user_id] = data
    
    if len(valid_users) == 0:
        return 0.0, {
            'total_users': len(user_data),
            'valid_users': 0,
            'total_samples': total_samples,
            'error': 'no_valid_users'
        }
    
    # 计算每个用户的AUC
    user_aucs = []
    for user_id, data in valid_users.items():
        if len(set(data['labels'])) > 1:  # 确保有正负样本
            user_auc = roc_auc_score(data['labels'], data['predictions'])
            user_aucs.append(user_auc)
    
    if len(user_aucs) == 0:
        return 0.0, {
            'total_users': len(user_data),
            'valid_users': len(valid_users),
            'total_samples': total_samples,
            'error': 'auc_calculation_failed'
        }
    
    # 计算GAUC（简单平均）
    gauc_score = np.mean(user_aucs)
    
    gauc_info = {
        'total_users': len(user_data),
        'valid_users': len(user_aucs),
        'total_samples': total_samples,
        'user_auc_mean': float(np.mean(user_aucs)),
        'user_auc_std': float(np.std(user_aucs))
    }
    
    if verbose:
        print(f"✅ GAUC: {gauc_score:.4f} (有效用户: {len(user_aucs)})")
    
    return gauc_score, gauc_info


def calculate_gauc(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    min_samples_per_user: int = 2,
    max_batches: Optional[int] = None,
    verbose: bool = True
) -> Tuple[float, Dict[str, Any]]:
    """计算GAUC（标准版本）"""
    if verbose:
        print("🔍 开始计算GAUC...")
    
    user_data = defaultdict(lambda: {'predictions': [], 'labels': []})
    total_samples = 0
    
    for batch_idx, (features, labels) in enumerate(dataset):
        if max_batches and batch_idx >= max_batches:
            break
            
        if "user_id" not in features:
            raise ValueError("数据集中缺少'user_id'字段")
        
        predictions = model(features, training=False)
        user_ids = features["user_id"].numpy()
        pred_values = predictions.numpy().flatten()
        label_values = labels.numpy().flatten()
        
        # 检查数组长度是否一致
        n_users = len(user_ids)
        n_preds = len(pred_values)
        n_labels = len(label_values)
        
        if n_users != n_preds or n_users != n_labels:
            if verbose:
                print(f"⚠️ 批次 {batch_idx} 数组长度不匹配: users={n_users}, preds={n_preds}, labels={n_labels}")
            # 取最小长度确保索引安全
            min_length = min(n_users, n_preds, n_labels)
            user_ids = user_ids[:min_length]
            pred_values = pred_values[:min_length]
            label_values = label_values[:min_length]
        
        # 按用户分组
        for i in range(len(user_ids)):
            user_id = user_ids[i]
            user_id_str = user_id.decode('utf-8') if isinstance(user_id, bytes) else str(user_id)
            user_data[user_id_str]['predictions'].append(float(pred_values[i]))
            user_data[user_id_str]['labels'].append(int(label_values[i]))
        
        total_samples += len(user_ids)
    
    return _calculate_gauc_from_user_data(user_data, min_samples_per_user, total_samples, verbose)


def save_gauc_results(
    gauc_score: float,
    gauc_info: Dict[str, Any],
    output_dir: str = "logs"
) -> str:
    """保存GAUC结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gauc_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    results = {
        'timestamp': timestamp,
        'gauc_score': gauc_score,
        'gauc_info': gauc_info
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return filepath


def compare_auc_gauc(
    auc_score: float,
    gauc_score: float,
    verbose: bool = True
) -> Dict[str, Any]:
    """对比AUC和GAUC"""
    auc_gauc_diff = auc_score - gauc_score
    relative_diff = (auc_gauc_diff / auc_score) * 100 if auc_score > 0 else 0
    
    # 简化的解读
    if abs(relative_diff) < 1:
        interpretation = "AUC和GAUC基本一致"
    elif auc_gauc_diff > 0:
        if relative_diff > 10:
            interpretation = "AUC明显高于GAUC，可能存在用户偏差"
        else:
            interpretation = "AUC高于GAUC，属于正常范围"
    else:
        interpretation = "GAUC高于AUC"
    
    comparison = {
        'auc_score': float(auc_score),
        'gauc_score': float(gauc_score),
        'absolute_difference': float(auc_gauc_diff),
        'relative_difference_percent': float(relative_diff),
        'interpretation': interpretation
    }
    
    if verbose:
        print(f"📊 AUC: {auc_score:.4f}, GAUC: {gauc_score:.4f}")
        print(f"   差异: {auc_gauc_diff:.4f} ({relative_diff:.1f}%)")
        print(f"   解读: {interpretation}")
    
    return comparison


def validate_gauc_calculation(
    model: tf.keras.Model,
    small_dataset: tf.data.Dataset,
    verbose: bool = True
) -> bool:
    """验证GAUC计算正确性"""
    gauc_score, gauc_info = calculate_gauc(
        model, small_dataset, min_samples_per_user=1, verbose=False
    )
    
    # 基本检查
    valid = (
        0 <= gauc_score <= 1 and
        gauc_info.get('valid_users', 0) > 0 and
        gauc_info.get('total_samples', 0) > 0
    )
    
    if verbose:
        print(f"GAUC验证: {'✅ 通过' if valid else '❌ 失败'}")
    
    return valid 