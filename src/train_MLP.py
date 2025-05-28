#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLP推送二分类模型训练脚本
"""

import os
import sys
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

from light_ctr.utils.constants import TF_DTYPE_MAPPING # type: ignore

# 直接从专门的模块导入函数，而不是通过兼容层
from src.utils.environment_utils import setup_environment
from src.utils.training_utils import train_model
from src.utils.feature_analysis_utils import check_feature_importance, plot_feature_importance
from src.utils.config_loader import load_data_config, load_train_config

from src.data.dataset_utils import inspect_datasets
from src.data.data_preparation import prepare_datasets
from src.utils.gpu_utils import setup_gpu
from src.models.model_utils import create_and_compile_model, test_model_on_batch
from src.models.mlp import MLP


def set_random_seeds(seed=42):
    """设置随机种子以确保结果可复现"""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    print(f"随机种子已设置为: {seed}")


def setup_directories():
    """创建必要的日志和模型目录"""
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    print("已创建日志和模型目录")


def load_configurations():
    """加载所有必要的配置文件"""
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


def main():
    """主函数，执行训练流程"""
    # 1. 环境设置和配置
    setup_gpu()
    project_root = setup_environment()
    setup_directories()
    set_random_seeds()
    
    # 2. 加载配置
    data_config, train_config = load_configurations()
    
    # 3. 数据准备
    full_dataset, train_dataset, validation_dataset, column_names, input_signature = prepare_datasets(
        data_config, train_config, TF_DTYPE_MAPPING
    )
    
    # 4. 检查数据集
    inspect_datasets(full_dataset, train_dataset, validation_dataset)
    
    # 5. 创建并编译模型
    model = create_and_compile_model(MLP, train_config)
    
    # 6. 测试模型
    if not test_model_on_batch(model, full_dataset):
        print("跳过训练步骤")
        return
    
    # 7. 训练模型
    history = train_model(model, full_dataset, train_dataset, validation_dataset, train_config=train_config)
    
    # 8. 输出训练结果
    print(f"模型训练完成")
    print(f"训练集最终AUC: {history.history['auc'][-1]:.6f}")
    print(f"验证集最终AUC: {history.history['val_auc'][-1]:.6f}")
    
    # 9. 评估特征重要性
    print("\n开始评估特征重要性...")
    feature_importance = check_feature_importance(model, validation_dataset, train_config=train_config)
    
    # 10. 绘制特征重要性图
    plot_feature_importance(feature_importance)


if __name__ == "__main__":
    main() 