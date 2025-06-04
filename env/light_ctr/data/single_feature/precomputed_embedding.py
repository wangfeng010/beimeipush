from typing import Dict, Optional, Any

import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.layers import Layer


class PrecomputedEmbedding(Layer):
    """
    预先计算的embedding加载层。
    
    此层从预先计算的embedding文件中加载对应的向量表示。
    
    Attributes:
        config: 包含以下配置项的字典：
            embedding_dir (str): 预先计算的embedding目录
            embedding_dim (int): embedding向量的维度
            id_map_file (str): 可选，ID到索引的映射文件
            embedding_type (str): 嵌入类型，可选值为"title_content"、"push_title"、"push_content"
    """

    # 默认配置
    DEFAULT_CONFIG = {
        "embedding_dir": "data/precomputed_embeddings",
        "embedding_dim": 128,
        "id_map_file": None,
        "embedding_type": "title_content",  # 新增参数，支持不同类型的嵌入
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        初始化PrecomputedEmbedding层。

        Args:
            config: 配置参数字典
            **kwargs: 传递给父类的其他参数
        """
        # 首先调用父类的初始化方法，只传递keras能理解的参数
        super(PrecomputedEmbedding, self).__init__(**kwargs)
        
        # 使用默认配置，然后用提供的配置覆盖
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # 加载配置
        self.embedding_dir = self.config["embedding_dir"]
        self.embedding_dim = self.config["embedding_dim"]
        self.id_map_file = self.config["id_map_file"]
        self.embedding_type = self.config["embedding_type"]
        
        # 加载数据结构
        self.embeddings = {}
        self.id_map = {}
        self._load_embeddings()
        
        # 为未找到的embedding准备一个零向量
        self.zero_vector = tf.zeros([self.embedding_dim], dtype=tf.float32)
        
        # ✅ 新增：添加全局行计数器，用于跟踪当前处理到第几行  
        self.global_row_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

    def _load_embeddings(self):
        """加载所有预先计算的embeddings"""
        print(f"加载预先计算的embeddings，目录: {self.embedding_dir}，类型: {self.embedding_type}")
        
        # 检查embedding目录是否存在
        if not os.path.exists(self.embedding_dir):
            print(f"警告: embedding目录不存在: {self.embedding_dir}")
            return
        
        # 根据embedding_type选择对应的文件后缀
        embedding_suffix = f"{self.embedding_type}_embeddings.pkl"
        
        # 加载所有embedding文件
        for filename in os.listdir(self.embedding_dir):
            if filename.endswith(embedding_suffix):
                file_path = os.path.join(self.embedding_dir, filename)
                # 从文件名中提取数据集名称
                dataset_name = filename.split(f"_{self.embedding_type}_embeddings.pkl")[0]
                
                with open(file_path, 'rb') as f:
                    embeddings = pickle.load(f)
                    self.embeddings[dataset_name] = tf.convert_to_tensor(embeddings, dtype=tf.float32)
                
                print(f"加载了{self.embedding_type} embedding文件: {filename}, 形状: {embeddings.shape}")
        
        # 如果指定了ID映射文件，则加载它
        if self.id_map_file and os.path.exists(self.id_map_file):
            with open(self.id_map_file, 'rb') as f:
                self.id_map = pickle.load(f)

    def call(self, inputs):
        """执行层的前向传播"""
        
        def get_embedding_by_row_index(input_str, row_idx):
            """
            🚀 新方案：根据数据在CSV中的实际行位置获取对应的预计算嵌入
            
            Args:
                input_str: 时间戳字符串（用于确定数据集）  
                row_idx: 在当前批次中的行索引
                
            Returns:
                对应的BERT嵌入向量
            """
            try:
                # 从日期时间字符串中提取日期部分作为数据集名称
                if isinstance(input_str, tf.Tensor):
                    input_str = input_str.numpy().decode('utf-8')
                else:
                    input_str = input_str.decode('utf-8')
                
                # 提取日期部分，格式为YYYYMMDD
                date_parts = input_str.split(' ')[0].split('-')
                if len(date_parts) == 3:
                    dataset_name = f"{date_parts[0]}{date_parts[1]}{date_parts[2]}"
                else:
                    return self.zero_vector
                
                if dataset_name in self.embeddings:
                    dataset_embs = self.embeddings[dataset_name]
                    
                    # 🎯 关键改进：使用全局行计数器而不是时间戳哈希
                    # 这确保了每一行数据对应其在CSV中的实际位置
                    current_global_idx = self.global_row_counter.numpy()
                    embedding_idx = current_global_idx % dataset_embs.shape[0]
                    
                    # 更新全局计数器
                    self.global_row_counter.assign_add(1)
                    
                    return dataset_embs[embedding_idx]
                
            except Exception as e:
                print(f"错误处理输入 {input_str}: {str(e)}")
            
            return self.zero_vector
        
        # 🚀 批处理优化：预先计算批次大小并重置计数器
        batch_size = tf.shape(inputs)[0]
        
        def process_batch_with_indices(inputs_batch):
            """处理整个批次，每个样本使用其在批次中的位置"""
            results = []
            for i in tf.range(batch_size):
                input_str = inputs_batch[i]
                embedding = tf.py_function(
                    lambda: get_embedding_by_row_index(input_str, i),
                    [],
                    tf.float32
                )
                embedding.set_shape([self.embedding_dim])
                results.append(embedding)
            
            return tf.stack(results)
        
        # 执行批处理
        result = process_batch_with_indices(inputs)
        
        return result

    def get_config(self):
        """获取层的配置，用于序列化"""
        base_config = super(PrecomputedEmbedding, self).get_config()
        base_config.update({"config": self.config})
        return base_config


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    
    print("测试 PrecomputedEmbedding")
    
    # 创建测试数据
    os.makedirs("test_embeddings", exist_ok=True)
    test_embeddings = np.random.random((100, 128)).astype(np.float32)
    with open("test_embeddings/test_title_content_embeddings.pkl", 'wb') as f:
        pickle.dump(test_embeddings, f)
    
    # 创建模型
    input_layer = Input(shape=(), dtype=tf.string)
    embedding_layer = PrecomputedEmbedding(config={
        "embedding_dir": "test_embeddings",
        "embedding_dim": 128,
        "embedding_type": "title_content"
    })
    output = embedding_layer(input_layer)
    model = Model(inputs=input_layer, outputs=output)
    
    # 测试输入
    test_inputs = tf.constant(["test:0", "test:50", "unknown"])
    
    # 预测
    results = model.predict(test_inputs)
    print(f"输出形状: {results.shape}")
    print(f"第一个embedding: {results[0][:5]}...")  # 显示前5个值
    
    # 清理测试数据
    import shutil
    shutil.rmtree("test_embeddings") 