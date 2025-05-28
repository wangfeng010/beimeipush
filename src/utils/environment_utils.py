#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
环境设置工具函数
"""

import os


def setup_environment() -> str:
    """设置工作环境并返回当前工作目录"""
    print("当前工作目录:", os.getcwd())
    
    # 如果在demo文件夹中运行，切换到项目根目录
    if os.path.basename(os.getcwd()) == 'demo':
        os.chdir("../")
        print("切换后的工作目录:", os.getcwd())
    
    # 返回当前工作目录的绝对路径
    return os.path.abspath(os.getcwd()) 