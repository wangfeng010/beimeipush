#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于TensorFlow原生Embedding的特征处理管道模块
专门处理UniProcess输出的哈希整数特征
"""

from typing import Dict, List, Tuple, Any, Optional
import tensorflow as tf


class FeaturePipelineBuilder:
    """基于TensorFlow原生Embedding的特征处理管道构建器"""
    
    def __init__(self, verbose: bool = True):
        """初始化特征处理管道构建器
        
        Args:
            verbose: 是否输出详细日志
        """
        self.embedding_layers = {}
        self.pooling_layers = {}
        self.verbose = verbose
    
    def build_feature_pipelines(self, configs: List[Dict[str, Any]]) -> List[Tuple[str, List]]:
        """构建特征处理管道
        
        从配置文件中读取特征信息，为每个特征创建对应的处理器序列
        
        Args:
            configs: 特征配置列表，来自feat.yml
            
        Returns:
            特征处理管道列表 [(特征名, [处理器列表])]
        """
        pipelines = []
        
        for config in configs:
            feature_name, processors = self._build_single_pipeline(config)
            if feature_name and processors:
                pipelines.append((feature_name, processors))
        
        if self.verbose:
            self._log_pipeline_summary(pipelines)
        
        return pipelines
    
    def _build_single_pipeline(self, config: Dict[str, Any]) -> Tuple[Optional[str], List]:
        """构建单个特征的处理管道
        
        解析单个特征配置，创建对应的处理器序列
        
        Args:
            config: 单个特征的配置信息
            
        Returns:
            (特征名, 处理器列表)
        """
        feature_name = config.get('feat_name')
        feat_type = config.get('feat_type')
        
        if not feature_name or not feat_type:
            if self.verbose:
                print(f"WARNING: 跳过无效配置: {config}")
            return None, []
        
        processors = self._create_processors(config)
        return feature_name, processors
    
    def _create_processors(self, config: Dict[str, Any]) -> List[tf.keras.layers.Layer]:
        """根据特征类型创建对应的处理器
        
        根据特征类型(sparse/varlen_sparse/dense)创建不同的处理器序列:
        - sparse: 只需要Embedding层
        - varlen_sparse: 需要Embedding层 + 池化层
        - dense: 使用Lambda层直接传递
        
        Args:
            config: 特征配置信息
            
        Returns:
            处理器列表
        """
        feat_name = config['feat_name']
        feat_type = config['feat_type']
        vocab_size = config.get('vocabulary_size', 1000)
        embed_dim = config.get('embedding_dim', 8)
        
        if feat_type == 'sparse':
            return [self._create_embedding(feat_name, vocab_size, embed_dim, sparse=True)]
        
        elif feat_type == 'varlen_sparse':
            embedding = self._create_embedding(feat_name, vocab_size, embed_dim, sparse=False)
            pooling = self._create_pooling(feat_name)
            return [embedding, pooling]
        
        elif feat_type == 'dense':
            return [self._create_dense_processor(feat_name)]
        
        else:
            if self.verbose:
                print(f"WARNING: 不支持的特征类型: {feat_type} for {feat_name}")
            return []
    
    def _create_embedding(self, name: str, vocab_size: int, embed_dim: int, 
                         sparse: bool = True) -> tf.keras.layers.Embedding:
        """创建Embedding层
        
        为特征创建TensorFlow Embedding层，将哈希整数转换为密集向量
        
        Args:
            name: 特征名称
            vocab_size: 词汇表大小
            embed_dim: 嵌入维度
            sparse: 是否为稀疏特征(影响是否启用masking)
            
        Returns:
            TensorFlow Embedding层
        """
        embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            name=f'{name}_embedding',
            mask_zero=not sparse,  # 变长特征需要masking padding值
            embeddings_initializer='uniform'
        )
        
        self.embedding_layers[name] = embedding
        
        if self.verbose:
            mask_info = "无mask" if sparse else "有mask"
            print(f"CREATED: {name}: Embedding({vocab_size}, {embed_dim}) {mask_info}")
        
        return embedding
    
    def _create_pooling(self, name: str, pooling_type: str = 'mean') -> tf.keras.layers.Layer:
        """创建池化层
        
        为变长特征创建池化层，将变长序列聚合为固定长度向量
        支持的池化类型: mean(平均), max(最大), sum(求和)
        
        Args:
            name: 特征名称
            pooling_type: 池化类型
            
        Returns:
            池化层
        """
        pooling_map = {
            'mean': tf.keras.layers.GlobalAveragePooling1D,
            'max': tf.keras.layers.GlobalMaxPooling1D,
            'sum': lambda: tf.keras.layers.Lambda(
                lambda x: tf.reduce_sum(x, axis=1), name=f'{name}_sum_pooling'
            )
        }
        
        pooling_cls = pooling_map.get(pooling_type, pooling_map['mean'])
        pooling = pooling_cls(name=f'{name}_{pooling_type}_pooling')
        
        self.pooling_layers[name] = pooling
        
        if self.verbose:
            print(f"CREATED: {name}: {pooling_type.title()}Pooling")
        
        return pooling
    
    def _create_dense_processor(self, name: str) -> tf.keras.layers.Layer:
        """创建密集特征处理器
        
        为密集特征创建处理器，当前实现为直通(identity)函数
        可根据需要扩展为标准化、归一化等操作
        
        Args:
            name: 特征名称
            
        Returns:
            处理器层
        """
        processor = tf.keras.layers.Lambda(
            lambda x: x, name=f'{name}_dense_processor'
        )
        
        if self.verbose:
            print(f"CREATED: {name}: Dense处理器")
        
        return processor
    
    def _log_pipeline_summary(self, pipelines: List[Tuple[str, List]]):
        """打印管道构建摘要
        
        输出所有成功构建的特征管道信息
        
        Args:
            pipelines: 构建完成的管道列表
        """
        print(f"\nSUCCESS: 成功构建 {len(pipelines)} 个特征管道:")
        for feature_name, processors in pipelines:
            processor_names = [p.__class__.__name__ for p in processors]
            print(f"   {feature_name}: {' -> '.join(processor_names)}")


def process_feature_batch(features: Dict[str, tf.Tensor], 
                         pipelines: List[Tuple[str, List]], 
                         verbose: bool = True) -> List[tf.Tensor]:
    """处理一批特征数据
    
    将输入的哈希整数特征转换为embedding向量
    这是特征处理的核心函数，负责协调所有特征的处理
    
    Args:
        features: 特征字典，包含哈希整数数据
                 格式: {'feature_name': tensor_data}
        pipelines: 特征处理管道列表
        verbose: 是否输出详细日志
        
    Returns:
        处理后的embedding向量列表
    """
    outputs = []
    processed_features = []
    
    if verbose:
        print(f"\nPROCESSING: 开始处理 {len(features)} 个输入特征")
    
    for feature_name, processors in pipelines:
        if feature_name not in features:
            if verbose:
                print(f"WARNING: 特征 {feature_name} 不在输入中")
            continue
        
        try:
            # 处理单个特征
            feature_output = _process_single_feature(
                features[feature_name], processors, feature_name, verbose
            )
            
            outputs.append(feature_output)
            processed_features.append(feature_name)
            
        except Exception as e:
            if verbose:
                print(f"ERROR: 处理特征 {feature_name} 失败: {e}")
            continue
    
    if verbose:
        print(f"SUCCESS: 成功处理 {len(outputs)} 个特征: {processed_features}")
    
    return outputs


def _process_single_feature(feature_input: tf.Tensor, 
                           processors: List[tf.keras.layers.Layer],
                           feature_name: str = "unknown",
                           verbose: bool = True) -> tf.Tensor:
    """处理单个特征
    
    对单个特征应用处理器序列，实现哈希整数到embedding向量的转换
    
    Args:
        feature_input: 特征输入张量(哈希整数)
        processors: 处理器序列
        feature_name: 特征名称(用于日志)
        verbose: 是否输出详细日志
        
    Returns:
        处理后的特征张量(embedding向量)
    """
    processed = feature_input
    
    if verbose:
        print(f"   PROCESSING: {feature_name}: {processed.shape}", end="")
    
    for processor in processors:
        processed = processor(processed)
        if verbose:
            print(f" -> {processed.shape}", end="")
    
    if verbose:
        print()  # 换行
    
    return processed


def create_sample_data(batch_size: int = 2) -> Dict[str, tf.Tensor]:
    """创建测试用的示例数据
    
    生成模拟的UniProcess输出数据，用于测试特征处理管道
    
    Args:
        batch_size: 批次大小
        
    Returns:
        示例特征数据字典
    """
    return {
        # 稀疏特征: 单个哈希整数
        'country_hash': tf.constant([156, 89] * (batch_size // 2 + 1))[:batch_size],
        'push_title_hash': tf.constant([3, 7] * (batch_size // 2 + 1))[:batch_size],
        
        # 变长稀疏特征: 哈希整数列表(已padding)
        'user_watch_stk_code_hash': tf.constant([
            [456, 789, 234, 0, 0],
            [123, 456, 0, 0, 0]
        ] * (batch_size // 2 + 1))[:batch_size],
        
        'tag_id_hash': tf.constant([
            [11, 22, 0],
            [33, 44, 55]
        ] * (batch_size // 2 + 1))[:batch_size]
    }


def create_sample_config() -> List[Dict[str, Any]]:
    """创建测试用的示例配置
    
    生成模拟的feat.yml配置，用于测试特征处理管道
    
    Returns:
        示例配置列表
    """
    return [
        {
            'feat_name': 'country_hash',
            'feat_type': 'sparse',
            'vocabulary_size': 200,
            'embedding_dim': 8
        },
        {
            'feat_name': 'user_watch_stk_code_hash',
            'feat_type': 'varlen_sparse',
            'vocabulary_size': 1000,
            'embedding_dim': 8
        }
    ]


def test_pipeline():
    """测试特征处理管道
    
    完整的测试流程，验证特征处理管道的正确性
    """
    print("TEST: 测试特征处理管道")
    
    # 创建配置和数据
    config = create_sample_config()
    sample_data = create_sample_data()
    
    # 构建管道
    builder = FeaturePipelineBuilder(verbose=True)
    pipelines = builder.build_feature_pipelines(config)
    
    # 处理特征
    outputs = process_feature_batch(sample_data, pipelines, verbose=True)
    
    print(f"\nRESULT: 最终结果: {len(outputs)} 个embedding向量")
    for i, output in enumerate(outputs):
        print(f"   输出 {i+1}: {output.shape}")
    
    return outputs


if __name__ == "__main__":
    test_pipeline() 