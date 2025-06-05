#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试JSON处理器
"""

import json
from src.models.deep.processors.json_array_processor import StockCodeProcessor, TagProcessor, ImportanceTagProcessor

def test_json_parsing():
    """测试JSON解析功能"""
    
    # 测试数据
    test_item_code = '''[{"market":"169","score":0,"code":"HIMS","tagId":"U000012934","name":"Hims & Hers Health","type":0,"parentId":"US_ROBOT0f37d7fd3fca6a41"},{"market":"169","score":0,"code":"NVO","tagId":"U000002999","name":"Novo Nordisk","type":0,"parentId":"US_ROBOT0f37d7fd3fca6a41"}]'''
    
    test_item_tags = '''[{"score":0.7803922295570374,"tagId":"51510","name":"us_high_importance","type":4,"parentId":"US_ROBOT0f37d7fd3fca6a41"},{"score":0.2,"tagId":"57967","name":"Fusion","type":4,"parentId":"US_ROBOT0f37d7fd3fca6a41"}]'''
    
    print("🔍 测试JSON解析功能")
    
    # 1. 测试原始JSON解析
    print("\n=== 测试原始JSON解析 ===")
    try:
        data = json.loads(test_item_code)
        print(f"✅ 成功解析item_code，条目数: {len(data)}")
        for i, item in enumerate(data):
            print(f"  项目{i+1}: code={item.get('code')}, name={item.get('name')}, market={item.get('market')}")
    except Exception as e:
        print(f"❌ 解析失败: {e}")
    
    try:
        data = json.loads(test_item_tags)
        print(f"✅ 成功解析item_tags，条目数: {len(data)}")
        for i, item in enumerate(data):
            print(f"  项目{i+1}: name={item.get('name')}, score={item.get('score')}, tagId={item.get('tagId')}")
    except Exception as e:
        print(f"❌ 解析失败: {e}")
    
    # 2. 测试处理器内部逻辑
    print("\n=== 测试处理器内部逻辑 ===")
    
    # 创建StockCodeProcessor实例
    stock_processor = StockCodeProcessor()
    print(f"StockCodeProcessor配置:")
    print(f"  extract_fields: {stock_processor.extract_fields}")
    print(f"  max_items: {stock_processor.max_items}")
    print(f"  output_format: {stock_processor.output_format}")
    
    # 手动测试处理逻辑
    try:
        data = json.loads(test_item_code)
        filtered_items = stock_processor._filter_and_sort_items(data)
        print(f"  过滤后条目数: {len(filtered_items)}")
        result = stock_processor._extract_and_format(filtered_items)
        print(f"  处理结果: {result}")
    except Exception as e:
        print(f"❌ 处理失败: {e}")
    
    # 创建TagProcessor实例  
    tag_processor = TagProcessor()
    print(f"\nTagProcessor配置:")
    print(f"  extract_fields: {tag_processor.extract_fields}")
    print(f"  use_score_weighting: {tag_processor.use_score_weighting}")
    print(f"  score_threshold: {tag_processor.score_threshold}")
    
    # 手动测试处理逻辑
    try:
        data = json.loads(test_item_tags)
        filtered_items = tag_processor._filter_and_sort_items(data)
        print(f"  过滤后条目数: {len(filtered_items)}")
        result = tag_processor._extract_and_format(filtered_items)
        print(f"  处理结果: {result}")
    except Exception as e:
        print(f"❌ 处理失败: {e}")
    
    # 3. 测试重要性标签处理器
    print("\n=== 测试ImportanceTagProcessor ===")
    importance_processor = ImportanceTagProcessor()
    print(f"ImportanceTagProcessor配置:")
    print(f"  extract_fields: {importance_processor.extract_fields}")
    print(f"  score_threshold: {importance_processor.score_threshold}")
    
    try:
        data = json.loads(test_item_tags)
        filtered_items = importance_processor._filter_and_sort_items(data)
        print(f"  过滤后条目数: {len(filtered_items)}")
        result = importance_processor._extract_and_format(filtered_items)
        print(f"  处理结果: {result}")
    except Exception as e:
        print(f"❌ 处理失败: {e}")

if __name__ == "__main__":
    test_json_parsing() 