#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSON数组处理器 - 用于解析和提取JSON数组中的结构化信息
"""

import tensorflow as tf
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Union

class JsonArrayProcessor(tf.keras.layers.Layer):
    """
    JSON数组处理器，用于解析JSON数组并提取关键字段
    
    支持多种提取策略：
    1. 提取指定字段并拼接
    2. 按权重提取（基于score字段）
    3. 限制最大item数量
    4. 支持多种输出格式
    """
    
    def __init__(self, 
                 extract_fields: List[str] = None,
                 max_items: int = 5,
                 sep: str = "|",
                 field_sep: str = "#",
                 use_score_weighting: bool = False,
                 score_threshold: float = 0.0,
                 output_format: str = "concat",  # concat, list, weighted
                 fill_value: str = "NULL",
                 **kwargs):
        """
        初始化JSON数组处理器
        
        Args:
            extract_fields: 要提取的字段列表，如 ["code", "name"]
            max_items: 最大处理的item数量
            sep: item之间的分隔符
            field_sep: 字段之间的分隔符
            use_score_weighting: 是否使用score字段进行权重排序
            score_threshold: score阈值，低于此值的item会被过滤
            output_format: 输出格式 - concat(拼接), list(列表), weighted(加权)
            fill_value: 填充值
        """
        super().__init__(**kwargs)
        
        self.extract_fields = extract_fields or ["code", "name"]
        self.max_items = max_items
        self.sep = sep
        self.field_sep = field_sep
        self.use_score_weighting = use_score_weighting
        self.score_threshold = score_threshold
        self.output_format = output_format
        self.fill_value = fill_value
        
    def call(self, inputs):
        """处理输入的JSON字符串"""
        
        def parse_single_json_wrapper(json_str_tensor):
            """包装函数，处理tensor输入"""
            # 从tensor中提取字符串
            json_str = json_str_tensor.numpy().decode('utf-8') if hasattr(json_str_tensor, 'numpy') else str(json_str_tensor)
            
            try:
                # 处理缺失值
                if pd.isna(json_str) or json_str == self.fill_value or json_str == "":
                    return self._create_empty_output()
                
                # 解析JSON
                data = json.loads(json_str)
                if not isinstance(data, list) or len(data) == 0:
                    return self._create_empty_output()
                
                # 过滤和排序
                filtered_items = self._filter_and_sort_items(data)
                
                # 提取字段并格式化输出
                return self._extract_and_format(filtered_items)
                
            except Exception as e:
                # 解析失败时返回空输出
                return self._create_empty_output()
        
        # 使用tf.map_fn处理批次，避免作用域问题
        def process_single_input(input_str):
            """处理单个输入"""
            result = tf.py_function(
                func=parse_single_json_wrapper,
                inp=[input_str],
                Tout=tf.string
            )
            result.set_shape(())
            return result
        
        # 使用tf.map_fn处理整个批次
        result = tf.map_fn(
            process_single_input,
            inputs,
            fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.string),
            parallel_iterations=1  # 确保顺序处理
        )
        
        return result
    
    def _filter_and_sort_items(self, items: List[Dict]) -> List[Dict]:
        """过滤和排序items"""
        filtered = []
        
        for item in items:
            # 检查score阈值
            if self.use_score_weighting:
                score = item.get('score', 0)
                if isinstance(score, (int, float)) and score < self.score_threshold:
                    continue
            
            # 检查必需字段是否存在
            if any(field in item for field in self.extract_fields):
                filtered.append(item)
        
        # 按score排序（如果启用）
        if self.use_score_weighting:
            filtered.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # 限制数量
        return filtered[:self.max_items]
    
    def _extract_and_format(self, items: List[Dict]) -> str:
        """提取字段并格式化输出"""
        if not items:
            return self._create_empty_output()
        
        if self.output_format == "concat":
            return self._format_concat(items)
        elif self.output_format == "list":
            return self._format_list(items)
        elif self.output_format == "weighted":
            return self._format_weighted(items)
        else:
            return self._format_concat(items)  # 默认使用concat
    
    def _format_concat(self, items: List[Dict]) -> str:
        """拼接格式：field1#field2|field1#field2|..."""
        results = []
        
        for item in items:
            field_values = []
            for field in self.extract_fields:
                value = str(item.get(field, self.fill_value))
                # 清理值，移除特殊字符
                value = value.replace(self.sep, "").replace(self.field_sep, "")
                field_values.append(value)
            
            if any(v != self.fill_value for v in field_values):
                results.append(self.field_sep.join(field_values))
        
        return self.sep.join(results) if results else self.fill_value
    
    def _format_list(self, items: List[Dict]) -> str:
        """列表格式：返回第一个主要字段的值"""
        if not items:
            return self.fill_value
        
        primary_field = self.extract_fields[0]
        values = [str(item.get(primary_field, self.fill_value)) for item in items]
        values = [v for v in values if v != self.fill_value]
        
        return self.sep.join(values[:self.max_items]) if values else self.fill_value
    
    def _format_weighted(self, items: List[Dict]) -> str:
        """加权格式：field1#field2#score|..."""
        results = []
        
        for item in items:
            field_values = []
            for field in self.extract_fields:
                value = str(item.get(field, self.fill_value))
                value = value.replace(self.sep, "").replace(self.field_sep, "")
                field_values.append(value)
            
            # 添加score
            score = item.get('score', 0)
            field_values.append(str(score))
            
            if any(v != self.fill_value for v in field_values[:-1]):  # 排除score检查
                results.append(self.field_sep.join(field_values))
        
        return self.sep.join(results) if results else self.fill_value
    
    def _create_empty_output(self) -> str:
        """创建空输出"""
        if self.output_format == "concat":
            # 返回填充的字段组合
            empty_fields = [self.fill_value] * len(self.extract_fields)
            return self.field_sep.join(empty_fields)
        else:
            return self.fill_value
    
    def get_config(self):
        """获取配置"""
        config = super().get_config()
        config.update({
            'extract_fields': self.extract_fields,
            'max_items': self.max_items,
            'sep': self.sep,
            'field_sep': self.field_sep,
            'use_score_weighting': self.use_score_weighting,
            'score_threshold': self.score_threshold,
            'output_format': self.output_format,
            'fill_value': self.fill_value
        })
        return config


class StockCodeProcessor(JsonArrayProcessor):
    """专门用于处理股票代码的处理器"""
    
    def __init__(self, **kwargs):
        # 股票代码专用配置
        default_config = {
            'extract_fields': ['code', 'name', 'market'],
            'max_items': 3,
            'sep': '|',
            'field_sep': '#',
            'use_score_weighting': False,
            'output_format': 'concat',
            'fill_value': 'NULL'
        }
        default_config.update(kwargs)
        super().__init__(**default_config)


class TagProcessor(JsonArrayProcessor):
    """专门用于处理标签的处理器"""
    
    def __init__(self, **kwargs):
        # 标签专用配置
        default_config = {
            'extract_fields': ['name', 'tagId'],
            'max_items': 5,
            'sep': '|',
            'field_sep': '#',
            'use_score_weighting': True,
            'score_threshold': 0.1,
            'output_format': 'weighted',
            'fill_value': 'NULL'
        }
        default_config.update(kwargs)
        super().__init__(**default_config)


class ImportanceTagProcessor(JsonArrayProcessor):
    """专门用于提取重要性标签的处理器"""
    
    def __init__(self, **kwargs):
        # 重要性标签配置
        default_config = {
            'extract_fields': ['name'],
            'max_items': 3,
            'sep': '|',
            'field_sep': '#',
            'use_score_weighting': True,
            'score_threshold': 0.5,
            'output_format': 'list',
            'fill_value': 'NULL'
        }
        default_config.update(kwargs)
        super().__init__(**default_config)
        
    def _filter_and_sort_items(self, items: List[Dict]) -> List[Dict]:
        """重写过滤逻辑，只保留重要性相关的标签"""
        importance_keywords = ['importance', 'high', 'middle', 'low', 'priority']
        
        filtered = []
        for item in items:
            name = item.get('name', '').lower()
            if any(keyword in name for keyword in importance_keywords):
                score = item.get('score', 0)
                if isinstance(score, (int, float)) and score >= self.score_threshold:
                    filtered.append(item)
        
        # 按score排序
        filtered.sort(key=lambda x: x.get('score', 0), reverse=True)
        return filtered[:self.max_items] 