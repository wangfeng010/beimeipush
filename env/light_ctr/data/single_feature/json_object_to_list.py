#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSON对象列表提取器 - 从JSON数组中提取指定字段的值列表
"""

import tensorflow as tf
import json
from typing import List, Any

class JsonObjectToList(tf.keras.layers.Layer):
    """
    从JSON对象字符串中提取指定键的值列表
    
    例如：
    输入: '[{"code":"HIMS","name":"Health"},{"code":"NVO","name":"Novo"}]'
    key: "code"
    输出: ["HIMS", "NVO"]
    """
    
    def __init__(self, key: str, fail_value: str = "null", **kwargs):
        """
        初始化JSON对象列表提取器
        
        Args:
            key: 要提取的键名
            fail_value: 提取失败时返回的默认值
        """
        super().__init__(**kwargs)
        self.key = key
        self.fail_value = fail_value
        
    def call(self, inputs):
        """处理输入的JSON字符串"""
        
        def extract_json_field(json_str_tensor):
            """从JSON字符串中提取指定字段"""
            # 从tensor中提取字符串
            json_str = json_str_tensor.numpy().decode('utf-8') if hasattr(json_str_tensor, 'numpy') else str(json_str_tensor)
            
            try:
                # 解析JSON
                json_obj = json.loads(json_str)
                
                # 如果是列表，提取每个对象中的指定字段
                if isinstance(json_obj, list):
                    result = [str(item.get(self.key, self.fail_value)) for item in json_obj if isinstance(item, dict)]
                    return result if result else [self.fail_value]
                else:
                    return [self.fail_value]
                    
            except Exception as e:
                # 解析失败时返回默认值
                return [self.fail_value]
        
        # 使用tf.map_fn处理批次
        def process_single_input(input_str):
            """处理单个输入"""
            result = tf.py_function(
                func=extract_json_field,
                inp=[input_str],
                Tout=tf.string
            )
            # 设置输出形状（变长序列）
            result.set_shape([None])
            return result
        
        # 使用tf.map_fn处理整个批次
        result = tf.map_fn(
            process_single_input,
            inputs,
            fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.string),
            parallel_iterations=1
        )
        
        return result
    
    def get_config(self):
        """获取配置"""
        config = super().get_config()
        config.update({
            'key': self.key,
            'fail_value': self.fail_value
        })
        return config


class Padding(tf.keras.layers.Layer):
    """
    序列填充层 - 将变长序列填充到固定长度
    """
    
    def __init__(self, max_len: int, pad_value: str = "null", **kwargs):
        """
        初始化填充层
        
        Args:
            max_len: 最大长度
            pad_value: 填充值
        """
        super().__init__(**kwargs)
        self.max_len = max_len
        self.pad_value = pad_value
        
    def call(self, inputs):
        """对输入序列进行填充"""
        # 如果输入是RaggedTensor，转换为密集tensor
        if isinstance(inputs, tf.RaggedTensor):
            # 填充到指定长度
            padded = inputs.to_tensor(
                default_value=self.pad_value,
                shape=[None, self.max_len]
            )
        else:
            # 如果已经是密集tensor，检查是否需要填充或截断
            current_shape = tf.shape(inputs)
            current_len = current_shape[1]
            
            # 如果当前长度小于max_len，进行填充
            padding_needed = tf.maximum(0, self.max_len - current_len)
            paddings = [[0, 0], [0, padding_needed]]
            padded = tf.pad(inputs, paddings, constant_values=self.pad_value)
            
            # 如果当前长度大于max_len，进行截断
            padded = padded[:, :self.max_len]
        
        return padded
    
    def get_config(self):
        """获取配置"""
        config = super().get_config()
        config.update({
            'max_len': self.max_len,
            'pad_value': self.pad_value
        })
        return config 