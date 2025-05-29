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
