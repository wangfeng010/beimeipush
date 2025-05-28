#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
向后兼容模块 - 导入所有拆分后的工具函数
"""

# 从各拆分模块导入函数
from .environment_utils import setup_environment
from .config_utils import (
    load_data_config,
    extract_config_info,
    load_feature_config,
    load_train_config
)
from .training_utils import train_model
from .feature_analysis_utils import (
    check_feature_importance,
    plot_feature_importance
)

# 提示：该模块已被拆分，请使用专门的工具模块
import warnings
warnings.warn(
    "train_utils模块已被拆分为环境、配置、训练和特征分析四个模块。"
    "请使用 environment_utils, config_utils, training_utils, 和 feature_analysis_utils 代替。",
    DeprecationWarning,
    stacklevel=2
)
