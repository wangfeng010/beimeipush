#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU相关工具函数
"""

import tensorflow as tf


def setup_gpu():
    """配置GPU设置，启用内存增长以避免内存溢出"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"找到 {len(gpus)} 个GPU设备，已启用内存增长")
            return True
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")
            return False
    else:
        print("未找到GPU设备，将使用CPU")
        return False


def get_available_devices():
    """获取可用的设备列表"""
    devices = []
    
    # 检查GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for i, gpu in enumerate(gpus):
            devices.append(f"GPU:{i} ({gpu.name})")
    
    # 检查CPU
    cpus = tf.config.experimental.list_physical_devices('CPU')
    if cpus:
        devices.append(f"CPU ({cpus[0].name})")
    
    return devices 