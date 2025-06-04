#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
核心数据分析：user_propernoun的语义vs记忆效应
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict

def analyze_user_propernoun_effects():
    """分析user_propernoun的混合效应"""
    
    # 加载数据
    df = pd.read_csv('data/train/20250520.csv')
    df['is_click'] = (df['log_type'] == 'PC').astype(int)
    
    print("=" * 60)
    print("user_propernoun特征效应分析")
    print("=" * 60)
    
    # 基础统计
    has_propernoun = df[df['user_propernoun'].notna()]
    no_propernoun = df[df['user_propernoun'].isna()]
    
    print(f"\n=== 基础统计 ===")
    print(f"有user_propernoun: {len(has_propernoun)} 样本, 点击率: {has_propernoun['is_click'].mean():.4f}")
    print(f"无user_propernoun: {len(no_propernoun)} 样本, 点击率: {no_propernoun['is_click'].mean():.4f}")
    print(f"点击率差异: {has_propernoun['is_click'].mean() - no_propernoun['is_click'].mean():.4f}")
    
    # 提取实体并分析语义匹配
    def extract_entities(propernoun_str):
        if pd.isna(propernoun_str):
            return []
        return [item.split('#')[0].strip().lower() for item in propernoun_str.split('|')]
    
    def check_content_match(entities, title, content):
        text = (str(title) + " " + str(content)).lower()
        return any(entity in text for entity in entities)
    
    # 添加实体信息
    df['user_entities'] = df['user_propernoun'].apply(extract_entities)
    df['has_content_match'] = df.apply(
        lambda row: check_content_match(row['user_entities'], row['push_title'], row['push_content']), 
        axis=1
    )
    
    print(f"\n=== 语义匹配分析 ===")
    no_propernoun_rate = no_propernoun['is_click'].mean()
    has_propernoun_no_match = df[(df['user_propernoun'].notna()) & (~df['has_content_match'])]['is_click'].mean()
    has_propernoun_with_match = df[(df['user_propernoun'].notna()) & (df['has_content_match'])]['is_click'].mean()
    
    print(f"无propernoun用户点击率: {no_propernoun_rate:.4f}")
    print(f"有propernoun但无匹配: {has_propernoun_no_match:.4f}")
    print(f"有propernoun且有匹配: {has_propernoun_with_match:.4f}")
    
    # 效应分解
    memory_effect = has_propernoun_no_match - no_propernoun_rate
    semantic_effect = has_propernoun_with_match - has_propernoun_no_match
    
    print(f"\n=== 效应分解 ===")
    print(f"纯记忆效应: {memory_effect:.4f} ({memory_effect/(memory_effect+semantic_effect)*100:.1f}%)")
    print(f"语义匹配效应: {semantic_effect:.4f} ({semantic_effect/(memory_effect+semantic_effect)*100:.1f}%)")
    
    # 分析特定实体
    print(f"\n=== 特定实体分析 ===")
    
    # 提取实体用户映射
    entity_users = defaultdict(list)
    for idx, row in df[df['user_propernoun'].notna()].iterrows():
        user_id = row['user_id']
        entities = extract_entities(row['user_propernoun'])
        for entity in entities:
            entity_users[entity].append(user_id)
    
    # 分析高频实体
    all_entities = []
    for entities in df['user_entities']:
        all_entities.extend(entities)
    
    entity_counts = Counter(all_entities)
    top_entities = ['china', 'us', 'nvidia', 'trump', 'fed']
    
    for entity in top_entities:
        if entity in entity_users and len(entity_users[entity]) > 10:
            users_with_entity = entity_users[entity]
            entity_user_data = df[df['user_id'].isin(users_with_entity)]
            
            # 检查包含该实体的内容
            contains_entity = entity_user_data.apply(
                lambda row: entity.lower() in (str(row['push_title']) + " " + str(row['push_content'])).lower(),
                axis=1
            )
            
            if contains_entity.sum() > 0 and (~contains_entity).sum() > 0:
                related_clicks = entity_user_data[contains_entity]['is_click'].mean()
                unrelated_clicks = entity_user_data[~contains_entity]['is_click'].mean()
                
                print(f"\n实体'{entity}'用户({len(users_with_entity)}个):")
                print(f"  相关内容点击率: {related_clicks:.4f} (样本: {contains_entity.sum()})")
                print(f"  无关内容点击率: {unrelated_clicks:.4f} (样本: {(~contains_entity).sum()})")
                print(f"  语义匹配提升: {related_clicks - unrelated_clicks:.4f}")
    
    return {
        'total_memory_effect': memory_effect,
        'total_semantic_effect': semantic_effect,
        'memory_ratio': memory_effect/(memory_effect+semantic_effect),
        'semantic_ratio': semantic_effect/(memory_effect+semantic_effect)
    }

if __name__ == "__main__":
    results = analyze_user_propernoun_effects()
    
    print(f"\n=== 核心结论 ===")
    print(f"user_propernoun特征包含混合效应:")
    print(f"1. 记忆效应 (活跃用户识别): {results['memory_ratio']*100:.1f}%")
    print(f"2. 语义效应 (兴趣匹配): {results['semantic_ratio']*100:.1f}%")
    print(f"3. 改进方向: 减少记忆效应，保持语义效应") 