#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training utilities for push binary classification model
"""

import os
import sys
import time
import json
import datetime
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Any, Tuple


def setup_environment() -> str:
    """设置工作环境并返回当前工作目录"""
    print("当前工作目录:", os.getcwd())
    
    # 如果在demo文件夹中运行，切换到项目根目录
    if os.path.basename(os.getcwd()) == 'demo':
        os.chdir("../")
        print("切换后的工作目录:", os.getcwd())
    
    # 返回当前工作目录的绝对路径
    return os.path.abspath(os.getcwd())


def load_data_config(config_path: str = "config/data.yml") -> Dict:
    """加载数据配置"""
    with open(config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    return data_config


def extract_config_info(data_config: Dict, TF_DTYPE_MAPPING: Dict) -> Tuple[str, List[str], List[Any], List[str]]:
    """从配置中提取数据信息"""
    file_pattern = data_config["train_dir"] + "/*.csv"
    
    # 从配置文件中提取列名和默认值类型
    column_names = [list(c.keys())[0] for c in data_config["raw_data_columns"]]
    column_defaults = [TF_DTYPE_MAPPING[list(c.values())[0]] for c in data_config["raw_data_columns"]]
    
    # 从配置文件中提取标签列的名称
    label_columns = data_config['label_columns']
    
    return file_pattern, column_names, column_defaults, label_columns


def load_feature_config(config_path: str = "config/feat.yml", exclude_features: List[str] = None) -> List[Dict]:
    """
    加载特征配置
    
    参数:
    - config_path: 配置文件路径
    - exclude_features: 要排除的特征列表，例如['user_id']
    
    返回:
    - 处理后的特征管道配置列表
    """
    # 如果没有提供要排除的特征，设置默认值
    if exclude_features is None:
        exclude_features = ['user_id']  # 默认排除user_id特征
    
    with open(config_path, 'r') as f:
        feat_config = yaml.safe_load(f)
    
    # 深度复制管道配置，以便安全修改
    pipelines_copy = []
    excluded_pipelines = 0
    
    for pipeline in feat_config['pipelines']:
        # 检查这个管道是否处理的是被排除的特征
        should_exclude = False
        
        # 检查管道的第一个操作是否针对被排除的特征
        if 'operations' in pipeline and pipeline['operations']:
            first_op = pipeline['operations'][0]
            if 'col_in' in first_op and first_op['col_in'] in exclude_features:
                should_exclude = True
                excluded_pipelines += 1
                print(f"排除特征: {first_op['col_in']}")
                continue  # 跳过此管道
        
        # 如果不需要排除，则处理这个管道
        if not should_exclude:
            pipeline_copy = {}
            for key, value in pipeline.items():
                if key == 'operations':
                    operations_copy = []
                    for operation in value:
                        operation_copy = operation.copy()
                        # 如果是FillNaString操作，替换为CustomFillNaString
                        if operation_copy.get('func_name') == 'FillNaString':
                            operation_copy['func_name'] = 'CustomFillNaString'
                        operations_copy.append(operation_copy)
                    pipeline_copy[key] = operations_copy
                else:
                    pipeline_copy[key] = value
            pipelines_copy.append(pipeline_copy)
    
    print(f"特征过滤结果: 总共{len(feat_config['pipelines'])}个管道，排除了{excluded_pipelines}个管道，保留{len(pipelines_copy)}个管道")
    
    return pipelines_copy


def load_train_config(config_path: str = "config/train.yml") -> Dict:
    """加载训练配置"""
    with open(config_path, 'r') as f:
        train_config = yaml.safe_load(f)
    return train_config


def train_model(model, dataset, train_dataset, validation_dataset, train_config=None):
    """
    训练模型并保存
    
    参数:
    - model: 要训练的模型
    - dataset: 完整数据集（用于调试）
    - train_dataset: 训练数据集
    - validation_dataset: 验证数据集
    - train_config: 训练配置参数（可选）
    
    返回:
    - history: 训练历史
    """
    # 获取训练参数
    if train_config is None:
        # 默认值
        epochs = 2
        batch_size = 256
        lr = 0.0005
        weight_decay = 0.001
    else:
        epochs = train_config['training']['epochs'] 
        batch_size = train_config['training']['batch_size']
        lr = train_config['training']['lr']
        weight_decay = train_config['training']['weight_decay']
    
    # 定义回调函数
    callbacks = [
        # 模型检查点
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./models/push_binary_classification_model.keras",
            save_best_only=False,
            save_weights_only=False),
        # 保存最佳模型
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./models/best_model.keras",
            save_best_only=True,
            monitor='val_auc',
            mode='max',
            save_weights_only=False),
        # 早停
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=2,
            mode='max',
            restore_best_weights=True),
        # 学习率调度器
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=1,
            min_lr=0.00001,
            verbose=1),
        # CSV日志
        tf.keras.callbacks.CSVLogger(
            f'./logs/training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            append=False)
    ]
    
    # 创建日志目录
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 训练模型
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # 记录训练结束时间
    end_time = time.time()
    training_time = end_time - start_time
    print(f"训练完成，总时间: {training_time:.2f} 秒")
    
    # 获取最终的训练和验证指标
    final_train_auc = history.history['auc'][-1]
    final_val_auc = history.history['val_auc'][-1]
    
    print(f"最终训练AUC: {final_train_auc:.4f}")
    print(f"最终验证AUC: {final_val_auc:.4f}")
    print(f"AUC差距: {abs(final_train_auc - final_val_auc):.4f}")
    
    # 绘制训练过程中的AUC曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['auc'], label='训练 AUC')
    plt.plot(history.history['val_auc'], label='验证 AUC')
    plt.title('训练和验证 AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./logs/auc_curve_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    return history


def check_feature_importance(model, dataset, train_config=None):
    """
    使用排列重要性检查特征的重要性
    
    参数:
    - model: 已训练的模型
    - dataset: 测试数据集
    - train_config: 训练配置
    
    返回:
    - feature_importance: 特征重要性字典
    """
    # 设置参数
    num_batches = 5
    
    # 使用采样数据进行评估
    print("使用采样数据评估特征重要性...")
    print(f"使用 {num_batches} 个批次进行评估")
    sample_dataset = dataset.take(num_batches)
    
    # 获取原始性能作为基线
    all_labels = []
    all_preds = []
    
    # 收集预测和实际标签
    print("收集基线预测...")
    for i, (x, y) in enumerate(sample_dataset):
        try:
            # 使用模型进行预测
            preds = model(x)
            # 确保预测和标签是一维数组
            y_pred = preds.numpy().flatten()
            y_true = y.numpy().flatten()
            
            all_preds.extend(y_pred)
            all_labels.extend(y_true)
            print(f"已处理批次 {i+1}/{num_batches}")
        except Exception as e:
            print(f"处理批次 {i+1} 时出错: {e}")
    
    # 计算基线AUC
    baseline_auc = roc_auc_score(all_labels, all_preds)
    print(f"基线AUC: {baseline_auc:.4f}")
    
    # 获取所有特征名称
    first_batch = next(iter(sample_dataset))
    feature_names = list(first_batch[0].keys())
    
    # 对每个特征进行重要性评估
    feature_importance = {}
    
    for feature_name in feature_names:
        print(f"评估特征 '{feature_name}' 的重要性...")
        feature_aucs = []
        
        # 进行多次重复测试取平均值，以增加稳定性
        for repeat in range(3):
            # 使用特征替换方法
            all_preds_permuted = []
            all_labels_permuted = []
            
            # 遍历数据集
            for i, (x, y) in enumerate(sample_dataset):
                try:
                    # 创建x的副本
                    x_copy = {k: tf.identity(v) for k, v in x.items()}
                    
                    # 对特定特征进行随机化
                    if x_copy[feature_name].shape[0] > 1:  # 确保批次中有多个样本
                        x_copy[feature_name] = tf.random.shuffle(x_copy[feature_name])
                    
                    # 进行预测
                    preds = model(x_copy)
                    all_preds_permuted.extend(preds.numpy().flatten())
                    all_labels_permuted.extend(y.numpy().flatten())
                except Exception as e:
                    print(f"  预测错误: {e}")
                    continue
            
            # 如果收集到了足够的预测
            if len(all_preds_permuted) > 0:
                # 计算特征被随机化后的AUC
                permuted_auc = roc_auc_score(all_labels_permuted, all_preds_permuted)
                feature_aucs.append(permuted_auc)
        
        # 计算平均AUC
        if feature_aucs:
            avg_permuted_auc = np.mean(feature_aucs)
            # 特征重要性 = 基线AUC - 随机化后AUC
            importance = baseline_auc - avg_permuted_auc
            feature_importance[feature_name] = importance
            print(f"  特征 {feature_name} 重要性: {importance:.6f}")
        else:
            print(f"  无法评估特征 {feature_name} 的重要性")
            feature_importance[feature_name] = 0.0
    
    # 按重要性排序
    sorted_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True)}
    
    # 打印特征重要性
    print("\n特征重要性排名:")
    for i, (feature, importance) in enumerate(sorted_importance.items()):
        print(f"{i+1}. {feature}: {importance:.6f}")
    
    # 保存特征重要性到文件
    importance_file = f"./logs/feature_importance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(importance_file, 'w') as f:
        json.dump(sorted_importance, f, indent=2)
    
    print(f"特征重要性已保存到: {importance_file}")
    
    return sorted_importance


def plot_feature_importance(feature_importance):
    """绘制特征重要性图表"""
    # 按重要性排序
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features = [x[0] for x in sorted_features]
    importance = [x[1] for x in sorted_features]
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, importance, color='skyblue')
    plt.xlabel('重要性分数 (基线AUC - 随机化后AUC)')
    plt.ylabel('特征')
    plt.title('特征重要性排名')
    plt.tight_layout()
    plt.savefig(f'./logs/feature_importance_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    print("特征重要性图表已保存") 