#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
预先生成并保存BERT embeddings
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import tensorflow_text as text
from tqdm import tqdm
import pickle
import yaml
import glob
from pathlib import Path

# 加载配置
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 初始化BERT模型
def init_bert_model(model_url):
    print(f"初始化BERT模型: {model_url}")
    preprocessing = tf_hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        trainable=False, 
        name="preprocessing"
    )
    encoder = tf_hub.KerasLayer(
        model_url,
        trainable=False,
        name="encoder"
    )
    return preprocessing, encoder

# 生成BERT embedding
def generate_embedding(text_batch, preprocessing, encoder, max_seq_length=64):
    # 对空字符串进行处理
    text_batch = tf.strings.regex_replace(text_batch, "^$", "")
    
    # 预处理文本
    preprocessed = preprocessing(text_batch)
    
    # 获取BERT嵌入
    bert_outputs = encoder(preprocessed)
    
    # 使用CLS token的embedding作为整句表示
    return bert_outputs["pooled_output"].numpy()

# 批量处理数据，以减少内存使用
def process_in_batches(texts, preprocessing, encoder, batch_size=64, max_seq_length=64):
    total = len(texts)
    embeddings = []
    
    # 创建进度条
    with tqdm(total=total, desc="生成embeddings") as pbar:
        for i in range(0, total, batch_size):
            batch = texts[i:min(i+batch_size, total)]
            batch_embeddings = generate_embedding(batch, preprocessing, encoder, max_seq_length)
            embeddings.append(batch_embeddings)
            pbar.update(len(batch))
    
    # 合并所有批次的结果
    return np.concatenate(embeddings) if embeddings else np.array([])

# 主函数
def main():
    print("开始预先生成BERT embeddings")
    
    # 加载配置
    data_config = load_config("config/data.yml")
    feat_config = load_config("config/feat.yml")
    
    # 获取BERT配置
    bert_config = None
    for pipeline in feat_config.get("pipelines", []):
        if pipeline.get("feat_name") == "title_content_bert_cross":
            for operation in pipeline.get("operations", []):
                if operation.get("func_name") == "BertEmbedding":
                    bert_config = operation.get("func_parameters", {})
                    break
            if bert_config:
                break
    
    if not bert_config:
        print("错误：未找到BERT配置")
        return
    
    print(f"BERT配置: {bert_config}")
    
    # 初始化BERT模型
    preprocessing, encoder = init_bert_model(bert_config["model_url"])
    
    # 加载训练数据
    data_dir = data_config.get("train_dir", "data/train")
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print(f"错误：在 {data_dir} 中未找到CSV文件")
        return
    
    print(f"找到CSV文件: {len(csv_files)}个")
    
    # 创建预先计算的embedding目录
    embedding_dir = "data/precomputed_embeddings"
    os.makedirs(embedding_dir, exist_ok=True)
    
    # CSV格式设置
    csv_format = data_config.get("csv_format", {})
    separator = csv_format.get("separator", ",")
    header = csv_format.get("header", 0)
    
    print(f"使用CSV格式: 分隔符='{separator}', 表头={header}")
    
    # 创建ID映射字典
    id_map = {}
    
    # 为每个CSV文件生成embeddings
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        dataset_name = os.path.splitext(filename)[0]
        
        # 定义不同类型的embedding输出文件
        title_content_output_file = os.path.join(embedding_dir, f"{dataset_name}_title_content_embeddings.pkl")
        title_output_file = os.path.join(embedding_dir, f"{dataset_name}_title_embeddings.pkl")
        content_output_file = os.path.join(embedding_dir, f"{dataset_name}_content_embeddings.pkl")
        
        # 检查是否所有文件都已存在，如果存在则跳过
        if (os.path.exists(title_content_output_file) and 
            os.path.exists(title_output_file) and 
            os.path.exists(content_output_file)):
            print(f"跳过已存在的embedding文件: {dataset_name}")
            continue
        
        print(f"处理文件: {csv_file}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_file, sep=separator, header=header)
        
        # 为每一行添加唯一ID
        df["row_id"] = dataset_name + ":" + df.index.astype(str)
        
        # 填充空值
        df["push_title"] = df["push_title"].fillna("")
        df["push_content"] = df["push_content"].fillna("")
        
        # 合并标题和内容用于组合嵌入
        df["title_content"] = df["push_title"] + " " + df["push_content"]
        
        # 1. 生成标题+内容的组合embeddings
        if not os.path.exists(title_content_output_file):
            print(f"生成标题+内容的组合embeddings...")
            title_content_embeddings = process_in_batches(
            df["title_content"].values, 
            preprocessing, 
            encoder,
            batch_size=64,
            max_seq_length=bert_config.get("max_seq_length", 64)
        )
        
        # 保存embeddings
            with open(title_content_output_file, 'wb') as f:
                pickle.dump(title_content_embeddings, f)
        
            print(f"标题+内容embeddings已保存到: {title_content_output_file}")
            print(f"Embedding形状: {title_content_embeddings.shape}")
        
        # 2. 生成仅标题的embeddings
        if not os.path.exists(title_output_file):
            print(f"生成仅标题的embeddings...")
            title_embeddings = process_in_batches(
                df["push_title"].values, 
                preprocessing, 
                encoder,
                batch_size=64,
                max_seq_length=bert_config.get("max_seq_length", 64)
            )
            
            # 保存embeddings
            with open(title_output_file, 'wb') as f:
                pickle.dump(title_embeddings, f)
            
            print(f"标题embeddings已保存到: {title_output_file}")
            print(f"Embedding形状: {title_embeddings.shape}")
        
        # 3. 生成仅内容的embeddings
        if not os.path.exists(content_output_file):
            print(f"生成仅内容的embeddings...")
            content_embeddings = process_in_batches(
                df["push_content"].values, 
                preprocessing, 
                encoder,
                batch_size=64,
                max_seq_length=bert_config.get("max_seq_length", 64)
            )
            
            # 保存embeddings
            with open(content_output_file, 'wb') as f:
                pickle.dump(content_embeddings, f)
            
            print(f"内容embeddings已保存到: {content_output_file}")
            print(f"Embedding形状: {content_embeddings.shape}")
        
        # 更新ID映射
        for i, row_id in enumerate(df["row_id"].values):
            id_map[row_id] = {
                "dataset_name": dataset_name,
                "index": i
            }
    
    # 保存ID映射
    id_map_file = os.path.join(embedding_dir, "id_map.pkl")
    with open(id_map_file, 'wb') as f:
        pickle.dump(id_map, f)
    print(f"ID映射已保存到: {id_map_file}")
    
    print("预先生成BERT embeddings完成")

if __name__ == "__main__":
    main() 