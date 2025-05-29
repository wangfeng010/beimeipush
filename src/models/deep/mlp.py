#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLP Model for Push Notification
"""

from typing import Dict, List, Tuple, Any, Optional, Union

import tensorflow as tf

from src.models.deep.feature_pipeline import FeaturePipelineBuilder, process_feature_batch


class MLP(tf.keras.Model):
    """多层感知器模型，用于推送通知二分类"""

    def __init__(self, pipelines_config, train_config=None):
        super(MLP, self).__init__()
        
        # 创建特征处理管道
        pipeline_builder = FeaturePipelineBuilder()
        self.feature_pipelines = pipeline_builder.build_feature_pipelines(pipelines_config)
        
        # 创建特征连接层
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)
        
        # 获取模型配置参数
        layers, dropout_rates, l2_reg = self._get_model_params(train_config)
        
        # 创建分类器网络
        self.classifier = self._build_classifier(layers, dropout_rates, l2_reg)
    
    def _get_model_params(self, train_config: Optional[Dict[str, Any]]) -> Tuple[List[int], List[float], float]:
        """
        从训练配置中获取模型参数
        
        Args:
            train_config: 训练配置字典，如果为None则使用默认参数
            
        Returns:
            tuple: (layers, dropout_rates, l2_reg)
                layers: 隐藏层大小列表
                dropout_rates: Dropout比例列表
                l2_reg: L2正则化系数
        """
        # 默认模型参数
        default_layers = [64, 32]
        default_dropout_rates = [0.3, 0.3]
        default_l2_reg = 0.001
        
        # 如果没有配置或配置中没有模型部分，使用默认值
        if not train_config or 'model' not in train_config:
            return default_layers, default_dropout_rates, default_l2_reg
            
        # 从配置中读取参数，如果没有则使用默认值
        model_config = train_config['model']
        layers = model_config.get('layers', default_layers)
        dropout_rates = model_config.get('dropout_rates', default_dropout_rates)
        l2_reg = model_config.get('l2_regularization', default_l2_reg)
        
        return layers, dropout_rates, l2_reg
    
    def _build_classifier(self, layers, dropout_rates, l2_reg):
        """
        构建分类器网络
        
        参数:
            layers: 隐藏层大小列表
            dropout_rates: Dropout比例列表
            l2_reg: L2正则化系数
            
        返回:
            classifier: 分类器模型
        """
        classifier_layers = []
        
        # 添加批归一化层
        classifier_layers.append(tf.keras.layers.BatchNormalization())
        
        # 添加隐藏层
        for i, units in enumerate(layers):
            # 添加全连接层
            dense = tf.keras.layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            )
            classifier_layers.append(dense)
            
            # 添加Dropout层
            if i < len(dropout_rates):
                dropout = tf.keras.layers.Dropout(dropout_rates[i])
                classifier_layers.append(dropout)
        
        # 添加输出层
        output_layer = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        classifier_layers.append(output_layer)
        
        return tf.keras.Sequential(classifier_layers)
    
    def call(self, features, training=None):
        """模型前向传播"""
        # 处理所有特征
        processed_outputs = process_feature_batch(features, self.feature_pipelines)
        
        # 如果没有有效输出，报告错误
        if not processed_outputs:
            raise ValueError("没有可用的特征输出，请检查输入特征和特征处理管道")
        
        # 合并所有输出
        if len(processed_outputs) > 1:
            concat_outputs = self.concat_layer(processed_outputs)
        else:
            concat_outputs = processed_outputs[0]
        
        # 应用分类器
        predictions = self.classifier(concat_outputs)
        return predictions
    
    def train_step(self, data):
        """自定义训练步骤"""
        features, labels = data
        
        with tf.GradientTape() as tape:
            # 前向传播
            predictions = self(features, training=True)
            
            # 确保标签和预测形状一致
            labels = tf.expand_dims(labels, axis=1)
            
            # 计算损失
            loss = self._compute_loss(predictions, labels)
        
        # 计算梯度并应用
        self._apply_gradients(tape, loss)
        
        # 更新指标
        self.compiled_metrics.update_state(labels, predictions)
        
        # 返回包含损失和指标的字典
        return self._get_train_results(loss)
    
    def _compute_loss(self, predictions, labels):
        """
        计算模型损失
        
        参数:
            predictions: 模型预测值
            labels: 真实标签
            
        返回:
            total_loss: 总损失
        """
        # 计算每个样本的损失
        per_example_loss = tf.keras.losses.binary_crossentropy(labels, predictions)
        
        # 计算平均损失
        loss = tf.reduce_mean(per_example_loss)
        
        # 添加正则化损失
        regularization_loss = sum(self.losses) if self.losses else 0
        total_loss = loss + regularization_loss
        
        return total_loss
    
    def _apply_gradients(self, tape, loss):
        """
        计算并应用梯度
        
        参数:
            tape: 梯度带
            loss: 损失值
        """
        # 计算梯度
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # 应用梯度
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    
    def _get_train_results(self, loss):
        """
        获取训练结果
        
        参数:
            loss: 损失值
            
        返回:
            results: 训练结果字典
        """
        results = {'loss': loss}
        
        for metric in self.metrics:
            results[metric.name] = metric.result()
        
        return results 