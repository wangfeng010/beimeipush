#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
特征处理管道模块
"""

from typing import Dict, List, Tuple, Any, Optional, Union

import tensorflow as tf
from env.light_ctr.data.single_feature import SINGLE_PROCESSOR_DICT
from env.light_ctr.data.cross_feature import CROSS_PROCESSOR_DICT
from src.models.deep.processors import CustomFillNaString


class FeaturePipelineBuilder:
    """特征处理管道构建器"""
    
    def __init__(self):
        """初始化特征处理管道构建器"""
        self._init_processor_dicts()
    
    def _init_processor_dicts(self):
        """初始化特征处理器字典"""
        # 为模型注册自定义处理器
        custom_processors = {
            "CustomFillNaString": CustomFillNaString,
        }
        # 合并处理器字典 - JSON处理器现在已经在light_ctr框架中
        self.single_processor_dict = {**SINGLE_PROCESSOR_DICT, **custom_processors}
        self.cross_processor_dict = CROSS_PROCESSOR_DICT
    
    def build_feature_pipelines(self, pipelines_config: List[Dict[str, Any]]) -> List[Tuple[str, List]]:
        """
        构建特征处理管道
        
        参数:
            pipelines_config: 特征管道配置
            
        返回:
            feature_pipelines: 特征处理管道列表
        """
        feature_pipelines = []
        
        for pipeline in pipelines_config:
            feature_name, processors = self._build_single_pipeline(pipeline)
            
            # 只有当输入特征有效且处理器不为空时，才添加到管道中
            if feature_name and processors:
                feature_pipelines.append((feature_name, processors))
        
        return feature_pipelines
    
    def _build_single_pipeline(self, pipeline: Dict[str, Any]) -> Tuple[Optional[str], List]:
        """
        构建单个特征管道
        
        参数:
            pipeline: 单个管道配置
            
        返回:
            feature_name: 特征名称
            processors: 处理器列表
        """
        # 从配置中获取输入特征名
        feature_name = None
        if 'operations' in pipeline and pipeline['operations']:
            first_op = pipeline['operations'][0]
            if 'col_in' in first_op:
                feature_name = first_op['col_in']
        
        processors = []
        
        # 构建处理管道
        for operation in pipeline.get('operations', []):
            processor = self._create_processor(operation, pipeline)
            if processor:
                processors.append(processor)
        
        return feature_name, processors
    
    def _create_processor(self, operation: Dict[str, Any], pipeline: Dict[str, Any]) -> Optional[tf.keras.layers.Layer]:
        """
        创建单个处理器
        
        参数:
            operation: 操作配置
            pipeline: 管道配置
            
        返回:
            processor: 处理器实例，如果创建失败则返回None
        """
        # 提取操作参数
        func_name = operation['func_name']
        func_parameters = operation.get('func_parameters', {})
        
        # 决定使用哪个处理器字典
        processor_dict = self._select_processor_dict(pipeline)
        
        # 添加安全检查
        if func_name not in processor_dict:
            print(
                f"警告：找不到处理器 {func_name}，"
                f"可用处理器: {list(processor_dict.keys())}"
            )
            return None
        
        # 如果是数值操作但输入可能是字符串，需要先添加转换层
        if func_name in ['Normalized', 'MinMaxScaler']:
            return tf.keras.layers.Lambda(
                lambda x: tf.strings.to_number(x, out_type=tf.float32),
                name=f"str_to_number_{operation.get('col_in', 'unknown')}"
            )
        
        # 特殊处理需要使用config参数的处理器
        if func_name == 'SplitEmbedding':
            return self._create_split_embedding_processor(func_parameters)
        elif func_name == 'EntityOnlyEmbedding':
            return self._create_entity_only_embedding_processor(func_parameters)
        elif func_name == 'BertEmbedding':
            return self._create_bert_embedding_processor(func_parameters)
        elif func_name == 'PrecomputedEmbedding':
            return self._create_precomputed_embedding_processor(func_parameters)
        
        # 创建处理器实例
        return processor_dict[func_name](**func_parameters)
    
    def _create_split_embedding_processor(self, parameters: Dict[str, Any]) -> tf.keras.layers.Layer:
        """
        创建SplitEmbedding处理器，将旧格式参数转换为config格式
        
        参数:
            parameters: 操作参数
            
        返回:
            processor: SplitEmbedding处理器实例
        """
        # 获取处理器类
        SplitEmbedding = self.single_processor_dict['SplitEmbedding']
        
        # 将参数转换为config字典格式
        config = {}
        for key, value in parameters.items():
            config[key] = value
        
        # 创建并返回处理器
        return SplitEmbedding(config=config)
    
    def _create_entity_only_embedding_processor(self, parameters: Dict[str, Any]) -> tf.keras.layers.Layer:
        """
        创建EntityOnlyEmbedding处理器，将参数转换为config格式
        
        参数:
            parameters: 操作参数
            
        返回:
            processor: EntityOnlyEmbedding处理器实例
        """
        # 获取处理器类
        EntityOnlyEmbedding = self.single_processor_dict['EntityOnlyEmbedding']
        
        # 将参数转换为config字典格式
        config = {}
        for key, value in parameters.items():
            config[key] = value
        
        # 创建并返回处理器
        return EntityOnlyEmbedding(config=config)
    
    def _create_bert_embedding_processor(self, parameters: Dict[str, Any]) -> tf.keras.layers.Layer:
        """
        创建BertEmbedding处理器，将参数转换为config格式
        
        参数:
            parameters: 操作参数
            
        返回:
            processor: BertEmbedding处理器实例
        """
        # 获取处理器类
        BertEmbedding = self.single_processor_dict['BertEmbedding']
        
        # 将参数转换为config字典格式
        config = {}
        for key, value in parameters.items():
            config[key] = value
        
        # 创建并返回处理器
        return BertEmbedding(config=config)
    
    def _create_precomputed_embedding_processor(self, parameters: Dict[str, Any]) -> tf.keras.layers.Layer:
        """
        创建PrecomputedEmbedding处理器，将参数转换为config格式
        
        参数:
            parameters: 操作参数
            
        返回:
            processor: PrecomputedEmbedding处理器实例
        """
        # 获取处理器类
        PrecomputedEmbedding = self.single_processor_dict['PrecomputedEmbedding']
        
        # 将参数转换为config字典格式
        config = {}
        for key, value in parameters.items():
            config[key] = value
        
        # 创建并返回处理器
        return PrecomputedEmbedding(config=config)
    
    def _select_processor_dict(self, pipeline: Dict[str, Any]) -> Dict[str, Any]:
        """
        选择适当的处理器字典
        
        参数:
            pipeline: 管道配置
            
        返回:
            processor_dict: 处理器字典
        """
        if pipeline['feat_type'] in ['SingleFeature', 'SparseFeature']:
            return self.single_processor_dict
        else:
            return self.cross_processor_dict


def process_feature_batch(features: Dict[str, tf.Tensor], 
                         feature_pipelines: List[Tuple[str, List]]) -> List[tf.Tensor]:
    """
    处理一批特征
    
    参数:
        features: 特征字典
        feature_pipelines: 特征处理管道列表
        
    返回:
        outputs: 处理后的特征输出列表
    """
    outputs = []
    used_features = []
    
    # 对每个特征管道应用处理
    for feature_name, processors in feature_pipelines:
        # 检查特征是否存在
        if feature_name in features:
            # 处理单个特征
            feature_output = process_single_feature(
                features[feature_name], 
                processors
            )
            
            # 记录使用的特征并添加到输出
            used_features.append(feature_name)
            outputs.append(feature_output)
    
    return outputs


def process_single_feature(feature_input: tf.Tensor, 
                          processors: List[tf.keras.layers.Layer]) -> tf.Tensor:
    """
    处理单个特征
    
    参数:
        feature_input: 特征输入张量
        processors: 处理器列表
        
    返回:
        processed_feature: 处理后的特征
    """
    processed = feature_input
    
    # 逐层应用处理
    for processor in processors:
        processed = processor(processed)
    
    return processed 