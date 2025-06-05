#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLP推送二分类模型训练脚本
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import tensorflow as tf

# 获取项目根目录和env目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
ENV_DIR = os.path.join(PROJECT_ROOT, 'env')

# 将必要的目录添加到Python路径
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if ENV_DIR not in sys.path:
    sys.path.insert(0, ENV_DIR)

# TensorFlow数据类型映射(去掉light_ctr依赖)
TF_DTYPE_MAPPING = {
    "float16": tf.float16,
    "float32": tf.float32,
    "float64": tf.float64,
    "int8": tf.int8,
    "int16": tf.int16,
    "int32": tf.int32,
    "int64": tf.int64,
    "uint8": tf.uint8,
    "uint16": tf.uint16,
    "uint32": tf.uint32,
    "uint64": tf.uint64,
    "bool": tf.bool,
    "string": tf.string,
    "complex64": tf.complex64,
    "complex128": tf.complex128,
    "qint8": tf.qint8,
    "qint16": tf.qint16,
    "qint32": tf.qint32,
    "quint8": tf.quint8,
    "quint16": tf.quint16,
}

# 导入所需的工具函数
from src.utils.environment_utils import setup_environment
from src.utils.training_utils import train_model
from src.utils.feature_analysis_utils import check_feature_importance, plot_feature_importance
from src.utils.config_loader import load_data_config, load_train_config
from src.data.dataset_utils import inspect_datasets
from src.data.data_preparation import prepare_datasets
from src.utils.gpu_utils import setup_gpu
from src.models.model_utils import create_and_compile_model, test_model_on_batch
# 从深度模型包导入MLP模型
from src.models.deep import MLP
from src.data.feature_preprocessor import apply_feature_preprocessing


def set_random_seeds(seed: int = 42) -> None:
    """
    设置随机种子以确保结果可复现
    
    参数:
        seed: 随机种子值
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print(f"随机种子已设置为: {seed}")


def setup_directories() -> None:
    """
    创建必要的日志和模型目录
    """
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    print("已创建日志和模型目录")


def load_configurations() -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    加载所有必要的配置文件
    
    返回:
        data_config: 数据配置
        train_config: 训练配置
    """
    # 加载数据配置
    data_config = load_data_config()
    print("数据配置已加载")
    
    # 加载训练配置
    try:
        train_config = load_train_config()
        print("训练配置已加载")
    except Exception as e:
        print(f"加载训练配置失败: {e}，将使用默认值")
        train_config = None
        
    return data_config, train_config


def trace_model(model: tf.keras.Model, dataset: tf.data.Dataset) -> None:
    """
    通过执行一次前向传递来追踪模型的所有函数，解决未追踪函数的警告问题
    
    参数:
        model: 需要追踪的模型
        dataset: 用于执行前向传递的数据集
    """
    # 获取一个批次的数据
    for batch in dataset.take(1):
        # 提取特征和标签
        features, labels = batch
        
        # 检查features是否为字典类型，这是多输入特征的常见情况
        if isinstance(features, dict):
            # 使用tf.function装饰器，为字典类型的输入创建输入签名
            @tf.function
            def trace_forward(inputs):
                return model(inputs, training=False)
            
            # 执行前向传递来追踪所有函数
            _ = trace_forward(features)
        else:
            # 单一输入特征的情况（不太可能出现在此模型中）
            @tf.function(input_signature=[tf.TensorSpec(shape=features.shape, dtype=features.dtype)])
            def trace_forward(inputs):
                return model(inputs, training=False)
            
            # 执行前向传递来追踪所有函数
            _ = trace_forward(features)
        
        print("模型函数追踪完成，这将减少保存模型时的未追踪函数警告")
        break


def prepare_model_and_data() -> Tuple[
    tf.keras.Model,
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset
]:
    """
    准备模型和数据
    
    返回:
        model: 编译好的模型
        full_dataset: 完整数据集
        train_dataset: 训练数据集
        validation_dataset: 验证数据集
    """
    # 1. 加载配置
    data_config, train_config = load_configurations()
    
    # 2. 数据准备
    datasets = prepare_dataset_from_config(data_config, train_config)
    full_dataset, train_dataset, validation_dataset = datasets[:3]
    
    # 3. 检查数据集
    inspect_datasets(full_dataset, train_dataset, validation_dataset)
    
    # 4. 创建并编译模型
    model = create_and_compile_model(MLP, train_config)
    
    # 5. 通过执行一次前向传递来追踪模型函数，减少保存时的警告
    trace_model(model, full_dataset)
    
    return model, full_dataset, train_dataset, validation_dataset


def prepare_dataset_from_config(
    data_config: Dict[str, Any], 
    train_config: Optional[Dict[str, Any]]
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str], Dict[str, tf.TensorShape]]:
    """
    从配置中准备数据集
    
    参数:
        data_config: 数据配置
        train_config: 训练配置
        
    返回:
        processed_full_dataset: 处理后的完整数据集
        processed_train_dataset: 处理后的训练数据集  
        processed_validation_dataset: 处理后的验证数据集
        column_names: 原始列名列表
        input_signature: 输入签名
    """
    print("\n" + "="*50)
    print("开始数据集准备和特征处理")
    print("="*50)
    
    # 1. 加载原始数据集
    print("\n📂 步骤1: 加载原始CSV数据...")
    full_dataset, train_dataset, validation_dataset, column_names, input_signature = prepare_datasets(
        data_config, train_config, TF_DTYPE_MAPPING
    )
    
    print(f"✅ 原始数据加载完成")
    print(f"   原始特征列: {column_names}")
    
    # 2. 应用特征处理
    print("\n🔧 步骤2: 应用UniProcess特征处理...")
    try:
        # 处理完整数据集
        print("   处理完整数据集...")
        processed_full_dataset = apply_feature_preprocessing(
            full_dataset, 
            feat_config_path="config/feat.yml",
            verbose=True
        )
        
        # 处理训练数据集
        print("   处理训练数据集...")
        processed_train_dataset = apply_feature_preprocessing(
            train_dataset,
            feat_config_path="config/feat.yml", 
            verbose=False  # 避免重复日志
        )
        
        # 处理验证数据集
        print("   处理验证数据集...")
        processed_validation_dataset = apply_feature_preprocessing(
            validation_dataset,
            feat_config_path="config/feat.yml",
            verbose=False  # 避免重复日志
        )
        
        print("✅ 特征处理完成")
        
        # 3. 验证处理后的数据集
        print("\n🔍 步骤3: 验证处理后的数据集...")
        _validate_processed_datasets(
            processed_full_dataset, 
            processed_train_dataset, 
            processed_validation_dataset
        )
        
        return (processed_full_dataset, processed_train_dataset, 
                processed_validation_dataset, column_names, input_signature)
        
    except Exception as e:
        print(f"❌ 特征处理失败: {e}")
        print("🔄 回退到原始数据集...")
        return full_dataset, train_dataset, validation_dataset, column_names, input_signature


def _validate_processed_datasets(full_dataset: tf.data.Dataset,
                                train_dataset: tf.data.Dataset, 
                                validation_dataset: tf.data.Dataset) -> None:
    """验证处理后的数据集
    
    Args:
        full_dataset: 处理后的完整数据集
        train_dataset: 处理后的训练数据集
        validation_dataset: 处理后的验证数据集
    """
    try:
        # 检查处理后的特征名称和数据类型
        for batch_features, batch_labels in full_dataset.take(1):
            print(f"   处理后特征数量: {len(batch_features)}")
            print(f"   特征名称: {list(batch_features.keys())}")
            
            # 检查前几个特征的数据类型和样例
            feature_sample = {}
            for i, (name, tensor) in enumerate(batch_features.items()):
                if i < 3:  # 只显示前3个特征的详细信息
                    feature_sample[name] = {
                        'shape': tensor.shape,
                        'dtype': tensor.dtype,
                        'sample_values': tensor.numpy()[:3] if tensor.shape[0] > 0 else 'empty'
                    }
                    print(f"   特征 '{name}': shape={tensor.shape}, dtype={tensor.dtype}")
            
            # 检查标签
            print(f"   标签shape: {batch_labels.shape}, dtype: {batch_labels.dtype}")
            
            break
            
        print("✅ 数据集验证通过")
        
    except Exception as e:
        print(f"⚠️  数据集验证警告: {e}")


def train_and_evaluate_model(
    model: tf.keras.Model,
    full_dataset: tf.data.Dataset,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    train_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    训练和评估模型
    
    参数:
        model: 待训练的模型
        full_dataset: 完整数据集
        train_dataset: 训练数据集
        validation_dataset: 验证数据集
        train_config: 训练配置
    """
    # 1. 测试模型
    if not test_model_on_batch(model, full_dataset):
        print("模型测试失败，跳过训练步骤")
        return
    
    # 2. 训练模型
    print("\n开始训练模型...")
    history = train_model(
        model, full_dataset, train_dataset, validation_dataset, 
        train_config=train_config
    )
    
    # 3. 输出训练结果
    print_training_results(history)
    
    # 打印模型的特征管道信息
    print("\n模型特征管道信息:")
    if hasattr(model, 'feature_pipelines'):
        print(f"共有 {len(model.feature_pipelines)} 个特征处理管道")
        for idx, (feature_name, processors) in enumerate(model.feature_pipelines):
            processor_names = [p.__class__.__name__ for p in processors]
            print(f"管道 #{idx+1}: 输入特征 '{feature_name}' -> 处理器: {' -> '.join(processor_names)}")
            
            # 特别标记BERT相关特征管道
            for processor in processors:
                if any(bert_class in processor.__class__.__name__ for bert_class in ['BertEmbedding', 'PrecomputedEmbedding']):
                    print(f"  [高级特征] 发现BERT/预计算处理器: {processor.__class__.__name__}")
    else:
        print("模型没有定义特征处理管道")
    
    # 4. 评估特征重要性
    print("\n开始评估特征重要性...")
    feature_importance = check_feature_importance(
        model, validation_dataset, train_config=train_config
    )
    
    # 5. 绘制特征重要性图
    plot_feature_importance(feature_importance)


def print_training_results(history: tf.keras.callbacks.History) -> None:
    """
    打印训练结果
    
    参数:
        history: 训练历史对象
    """
    print(f"\n模型训练完成")
    print(f"训练集最终AUC: {history.history['auc'][-1]:.6f}")
    print(f"验证集最终AUC: {history.history['val_auc'][-1]:.6f}")
    
    # 计算并打印过拟合指标
    train_auc = history.history['auc'][-1]
    val_auc = history.history['val_auc'][-1]
    auc_diff = train_auc - val_auc
    
    if auc_diff > 0.05:
        print(f"警告: 可能存在过拟合现象 (训练AUC - 验证AUC = {auc_diff:.4f})")


def setup_environment_for_training() -> None:
    """设置训练环境"""
    setup_gpu()
    setup_environment()
    setup_directories()
    set_random_seeds()
    print("训练环境设置完成")


def main() -> None:
    """主函数，执行训练流程"""
    # 1. 环境设置
    setup_environment_for_training()
    
    # 2. 准备模型和数据
    model, full_dataset, train_dataset, validation_dataset = prepare_model_and_data()
    
    # 3. 重新加载训练配置（确保使用最新配置）
    _, train_config = load_configurations()
    
    # 4. 训练和评估模型
    train_and_evaluate_model(
        model, full_dataset, train_dataset, validation_dataset, 
        train_config=train_config
    )
    
    print("\n训练流程完成")


if __name__ == "__main__":
    main() 