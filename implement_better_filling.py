#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进的user_propernoun填充策略核心实现
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Any, Optional

class ImprovedEntityEmbeddingFilling(tf.keras.layers.Layer):
    """改进的user_propernoun填充策略"""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.noise_scale = config.get('noise_scale', 0.1)
        self.embedding_dim = config.get('embedding_dim', 16)
        
    def build(self, input_shape):
        super().build(input_shape)
        
        # 初始化统计变量（从预计算的统计文件加载）
        self.mean_embedding = tf.Variable(
            tf.zeros([self.embedding_dim]), 
            trainable=False, 
            name='propernoun_mean'
        )
        self.std_embedding = tf.Variable(
            tf.ones([self.embedding_dim]) * 0.1, 
            trainable=False, 
            name='propernoun_std'
        )
    
    def call(self, inputs, training=None):
        """改进的填充策略"""
        # 检测缺失值（零向量）
        is_missing = tf.reduce_all(tf.equal(inputs, 0.0), axis=-1, keepdims=True)
        
        # 生成改进的填充值
        if training:
            # 训练时添加噪音：mean + N(0, noise_scale * std)
            noise = tf.random.normal(
                tf.shape(inputs), 
                mean=0.0, 
                stddev=self.noise_scale
            ) * self.std_embedding
            filled_values = self.mean_embedding + noise
        else:
            # 推理时使用平均值
            filled_values = tf.tile(
                tf.expand_dims(self.mean_embedding, 0),
                [tf.shape(inputs)[0], 1]
            )
        
        # 条件填充：缺失时用新值，否则保持原值
        outputs = tf.where(is_missing, filled_values, inputs)
        return outputs
    
    def set_statistics(self, mean_emb: np.ndarray, std_emb: np.ndarray):
        """设置统计量（在训练前调用）"""
        self.mean_embedding.assign(mean_emb)
        self.std_embedding.assign(std_emb)

def compute_propernoun_statistics(data_path: str, output_path: str):
    """计算user_propernoun的统计量"""
    import json
    
    # 加载数据
    df = pd.read_csv(data_path)
    
    # 提取所有有效的embedding
    def extract_embedding(propernoun_str: str) -> Optional[np.ndarray]:
        if pd.isna(propernoun_str) or propernoun_str == "NULL#0":
            return None
            
        try:
            entities_scores = []
            for item in propernoun_str.split('|'):
                parts = item.split('#')
                if len(parts) == 2:
                    entity = parts[0].strip().lower()
                    score = float(parts[1])
                    entities_scores.append((entity, score))
            
            if not entities_scores:
                return None
            
            # 模拟EntityOnlyEmbedding的处理逻辑
            embedding = np.zeros(16)
            for entity, score in entities_scores:
                entity_hash = abs(hash(entity)) % 16
                embedding[entity_hash] += score
            
            # 归一化
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
                
            return None
        except:
            return None
    
    embeddings = []
    for propernoun in df['user_propernoun'].dropna():
        emb = extract_embedding(propernoun)
        if emb is not None:
            embeddings.append(emb)
    
    if not embeddings:
        raise ValueError("没有找到有效的propernoun embedding")
    
    embeddings = np.array(embeddings)
    
    # 计算统计量
    mean_emb = np.mean(embeddings, axis=0)
    std_emb = np.std(embeddings, axis=0)
    std_emb = np.maximum(std_emb, 0.01)  # 避免标准差为0
    
    # 保存统计量
    stats = {
        'mean_embedding': mean_emb.tolist(),
        'std_embedding': std_emb.tolist(),
        'num_samples': len(embeddings),
        'embedding_dim': len(mean_emb)
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ 统计量已保存到 {output_path}")
    print(f"   样本数: {len(embeddings)}")
    print(f"   平均值模长: {np.linalg.norm(mean_emb):.4f}")
    print(f"   平均标准差: {np.mean(std_emb):.4f}")

if __name__ == "__main__":
    # 使用示例
    compute_propernoun_statistics(
        "data/train/20250520.csv",
        "config/propernoun_stats.json"
    ) 