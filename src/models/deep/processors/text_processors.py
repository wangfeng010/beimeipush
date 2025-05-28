#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文本特征处理器模块
"""

import tensorflow as tf


class CustomFillNaString(tf.keras.layers.Layer):
    """
    对输入特征进行空字符串的填充，适用于CSV数据。
    """

    def __init__(self, fill_value="0", **kwargs):
        super(CustomFillNaString, self).__init__(**kwargs)
        self.fill_value = fill_value  # 用于填充空字符串的值

    def call(self, inputs):
        # 确保输入是字符串类型
        inputs = tf.strings.as_string(inputs)
        
        # 使用tf.where和tf.equal来替换空字符串
        condition = tf.equal(inputs, "")
        # 将条件为True的位置替换为fill_value
        filled_output = tf.where(condition, self.fill_value, inputs)
        return filled_output

    def get_config(self):
        config = super(CustomFillNaString, self).get_config()
        config.update({"fill_value": self.fill_value})
        return config 