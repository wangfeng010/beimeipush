from typing import Dict, Optional, Any

import tensorflow as tf
from tensorflow.keras.layers import Layer

from .split_transform import SplitTransform
from .str_embedding import StrEmbedding


class SplitEmbedding(Layer):
    """
    字符串预处理层，支持两次字符串拆分和嵌入向量生成。

    该层首先使用 `SplitTransform` 对输入字符串进行拆分，然后使用 `StrEmbedding` 
    对拆分后的字符串进行嵌入向量生成。支持对嵌入向量进行池化操作（如均值池化或最大池化）
    以生成固定长度的向量表示。

    Attributes:
        config: 包含以下配置项的字典：
        first_sep (str): 第一次拆分使用的分隔符，默认为 "|"。
        second_sep (str): 第二次拆分使用的分隔符，默认为 "#"。
        second_sep_pos (int): 第二次拆分的位置，默认为 0。
        padding_value (str): 用于填充的值，默认为 "PAD"。
        second_sep_item_num (int): 第二次拆分后保留的项数，默认为 2。
        max_length (int): 拆分后的最大长度，默认为 5。
        embedding_dim (int): 嵌入向量的维度，默认为 10。
        vocab_size (int): 词汇表的大小，默认为 1000。
        pooling (Optional[str]): 池化方法，可选值为 "mean"、"max" 或 None，默认为 "mean"。
    """

    # 默认配置
    DEFAULT_CONFIG = {
        "first_sep": "|",
        "second_sep": "#",
        "second_sep_pos": 0,
        "padding_value": "PAD",
        "second_sep_item_num": 2,
        "max_length": 5,
        "embedding_dim": 10,
        "vocab_size": 1000,
        "pooling": "mean",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        初始化SplitEmbedding层。

        Args:
            config: 配置参数字典，包含分隔符、嵌入维度等信息
            **kwargs: 传递给父类的其他参数
        """
        super(SplitEmbedding, self).__init__(**kwargs)
        
        # 使用默认配置，然后用提供的配置覆盖
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        # 为了向后兼容和方便访问，将配置项作为类属性
        self.first_sep = self.config["first_sep"]
        self.second_sep = self.config["second_sep"]
        self.second_sep_pos = self.config["second_sep_pos"]
        self.padding_value = self.config["padding_value"]
        self.max_length = self.config["max_length"]
        self.embedding_dim = self.config["embedding_dim"]
        self.vocab_size = self.config["vocab_size"]
        self.second_sep_item_num = self.config["second_sep_item_num"]
        self.pooling = self.config["pooling"]
        
        # 初始化词汇表相关变量
        self.vocab = set([self.padding_value])
        self.word_to_idx = {self.padding_value: 0}
        
        # 将在build方法中初始化的变量
        self.split_transform = None
        self.varlen_emb = None

    def build(self, input_shape: tf.TensorShape):
        """构建层，初始化需要的转换层和嵌入层"""
        # 初始化转换层
        transform_config = {
            "first_sep": self.first_sep,
            "second_sep": self.second_sep,
            "second_sep_pos": self.second_sep_pos,
            "padding_value": self.padding_value,
            "second_sep_item_num": self.second_sep_item_num,
            "max_length": self.max_length,
        }
        self.split_transform = SplitTransform(**transform_config)
        
        # 初始化嵌入层
        embedding_config = {
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.vocab_size,
            "padding_value": self.padding_value,
            "pooling": self.pooling,
        }
        self.varlen_emb = StrEmbedding(**embedding_config)
        
        super(SplitEmbedding, self).build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """执行层的前向传播"""
        extracted_data = self.split_transform(inputs)
        embedded_data = self.varlen_emb(extracted_data)
        return embedded_data

    def get_config(self):
        """获取层的配置，用于序列化"""
        base_config = super(SplitEmbedding, self).get_config()
        base_config.update({"config": self.config})
        return base_config


class EntityOnlyEmbedding(Layer):
    """
    实体词嵌入处理器 - 只保留实体名词，忽略数值信息
    
    与SplitEmbedding类似，但在处理格式如: "china#8.16|nvidia#3.06|u.s.#4.08"时，
    只提取实体词部分(china, nvidia, u.s.)，完全忽略数值信息(8.16, 3.06, 4.08)
    
    Attributes:
        config: 包含以下配置项的字典：
        first_sep (str): 第一次拆分使用的分隔符，默认为 "|"。
        second_sep (str): 第二次拆分使用的分隔符，默认为 "#"。
        padding_value (str): 用于填充的值，默认为 "PAD"。
        max_length (int): 拆分后的最大长度，默认为 10。
        embedding_dim (int): 嵌入向量的维度，默认为 16。
        vocab_size (int): 词汇表的大小，默认为 3000。
        pooling (Optional[str]): 池化方法，可选值为 "mean"、"max" 或 None，默认为 "mean"。
    """

    # 默认配置
    DEFAULT_CONFIG = {
        "first_sep": "|",
        "second_sep": "#",
        "padding_value": "PAD",
        "max_length": 10,
        "embedding_dim": 16,
        "vocab_size": 3000,
        "pooling": "mean",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        初始化EntityOnlyEmbedding层。

        Args:
            config: 配置参数字典，包含分隔符、嵌入维度等信息
            **kwargs: 传递给父类的其他参数
        """
        super(EntityOnlyEmbedding, self).__init__(**kwargs)
        
        # 使用默认配置，然后用提供的配置覆盖
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        # 为了向后兼容和方便访问，将配置项作为类属性
        self.first_sep = self.config["first_sep"]
        self.second_sep = self.config["second_sep"]
        self.padding_value = self.config["padding_value"]
        self.max_length = self.config["max_length"]
        self.embedding_dim = self.config["embedding_dim"]
        self.vocab_size = self.config["vocab_size"]
        self.pooling = self.config["pooling"]
        
        # 初始化词汇表相关变量
        self.vocab = set([self.padding_value])
        self.word_to_idx = {self.padding_value: 0}
        
        # 将在build方法中初始化的变量
        self.embedding_layer = None

    def build(self, input_shape: tf.TensorShape):
        """构建层，初始化嵌入层"""
        # 初始化嵌入层
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=(self.padding_value == 0),
            name="entity_embedding"
        )
        
        super(EntityOnlyEmbedding, self).build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """执行层的前向传播"""
        def process_string(input_string):
            """处理单个字符串，提取实体词"""
            # 按第一个分隔符分割
            parts = tf.strings.split([input_string], self.first_sep).values
            
            # 对每个部分按第二个分隔符分割，只保留第一部分（实体名）
            entities = tf.strings.split(parts, self.second_sep).values[::2]  # 只取偶数索引（实体名）
            
            # 截断到最大长度
            entities = entities[:self.max_length]
            
            # 将实体名转换为哈希ID
            entity_ids = tf.strings.to_hash_bucket_fast(entities, self.vocab_size - 1) + 1
            
            # 填充到固定长度
            padded_length = self.max_length
            current_length = tf.shape(entity_ids)[0]
            padding_needed = padded_length - current_length
            
            # 创建填充
            padding = tf.zeros([padding_needed], dtype=tf.int64)
            padded_ids = tf.concat([entity_ids, padding], axis=0)
            
            return padded_ids
        
        # 对批量数据应用处理函数
        sequences = tf.map_fn(
            process_string,
            inputs,
            fn_output_signature=tf.TensorSpec([self.max_length], tf.int64),
            parallel_iterations=10
        )
        
        # 转换为int32（Embedding层需要）
        sequences = tf.cast(sequences, tf.int32)
        
        # 通过嵌入层
        embeddings = self.embedding_layer(sequences)
        
        # 应用池化
        if self.pooling == "mean":
            # 计算非零元素的均值（避免填充值的影响）
            mask = tf.not_equal(sequences, 0)
            mask = tf.cast(mask, tf.float32)
            mask_sum = tf.reduce_sum(mask, axis=1, keepdims=True)
            mask_sum = tf.where(tf.equal(mask_sum, 0), tf.ones_like(mask_sum), mask_sum)
            
            mask = tf.expand_dims(mask, -1)
            masked_embeddings = embeddings * mask
            pooled_embeddings = tf.reduce_sum(masked_embeddings, axis=1) / mask_sum
            return pooled_embeddings
        elif self.pooling == "max":
            # 最大池化
            mask = tf.not_equal(sequences, 0)
            mask = tf.expand_dims(mask, -1)
            mask = tf.cast(mask, tf.float32)
            
            # 对填充位置使用大的负值
            neg_mask = (1.0 - mask) * -1e9
            masked_embeddings = embeddings * mask + neg_mask
            
            return tf.reduce_max(masked_embeddings, axis=1)
        else:
            # 不池化，返回完整序列
            return embeddings

    def get_config(self):
        """获取层的配置，用于序列化"""
        base_config = super(EntityOnlyEmbedding, self).get_config()
        base_config.update({"config": self.config})
        return base_config


if __name__ == "__main__":
    from tensorflow.keras.layers import Dense, InputLayer
    from tensorflow.keras.models import Sequential

    print("testing SplitEmbedding")

    # 构建模型
    def build_model():
        """构建测试用模型"""
        model = Sequential([
            InputLayer(input_shape=(), dtype=tf.string),
            SplitEmbedding(),  # 使用默认配置
            Dense(1, activation="sigmoid"),
        ])
        return model

    # 测试数据
    input_data = tf.constant([
            "德邦证券,吕品#5|中信建投证券,卢昊#2",
            "国海证券,刘熹#2|东北证券,王凤华#1|中航证券,李蔚#1",
    ])
    labels = tf.constant([1, 0])  # 假设是二分类标签
    
    # 构建并编译模型
    model = build_model()
    model.compile(
        optimizer="adam", 
        loss="binary_crossentropy", 
        metrics=["accuracy"]
    )
    
    # 打印模型摘要并训练
    model.summary()
    model.fit(input_data, labels, epochs=2)
    
    # 测试预测
    predictions = model.predict(input_data)
    print("预测结果:", predictions)
    print("testing finished")

    
    print("\ntesting EntityOnlyEmbedding")
    
    # 构建使用EntityOnlyEmbedding的模型
    def build_entity_only_model():
        """构建测试用模型"""
        model = Sequential([
            InputLayer(input_shape=(), dtype=tf.string),
            EntityOnlyEmbedding(),  # 使用默认配置
            Dense(1, activation="sigmoid"),
        ])
        return model
    
    # 测试EntityOnlyEmbedding
    entity_model = build_entity_only_model()
    entity_model.compile(
        optimizer="adam", 
        loss="binary_crossentropy", 
        metrics=["accuracy"]
    )
    
    # 打印模型摘要并训练
    entity_model.summary()
    entity_model.fit(input_data, labels, epochs=2)
    
    # 测试预测
    entity_predictions = entity_model.predict(input_data)
    print("实体词模型预测结果:", entity_predictions)
    print("EntityOnlyEmbedding testing finished")
