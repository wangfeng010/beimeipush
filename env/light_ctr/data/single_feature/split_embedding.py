from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer

from .split_transform import SplitTransform
from .str_embedding import StrEmbedding


class SplitEmbedding(Layer):
    """
    字符串预处理层，支持两次字符串拆分和嵌入向量生成。

    该层首先使用 `SplitTransform` 对输入字符串进行拆分，然后使用 `StrEmbedding` 对拆分后的字符串进行嵌入向量生成。
    支持对嵌入向量进行池化操作（如均值池化或最大池化）以生成固定长度的向量表示。

    Attributes:
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

    def __init__(
        self,
        first_sep: str = "|",
        second_sep: str = "#",
        second_sep_pos: int = 0,
        padding_value: str = "PAD",
        second_sep_item_num: int = 2,
        max_length: int = 5,
        embedding_dim: int = 10,
        vocab_size: int = 1000,
        pooling: Optional[str] = "mean",
        **kwargs,
    ):
        super(SplitEmbedding, self).__init__(**kwargs)
        self.first_sep = first_sep
        self.second_sep = second_sep
        self.second_sep_pos = second_sep_pos
        self.padding_value = padding_value
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size  # 固定大小的词表
        self.vocab = set()
        self.vocab.add(self.padding_value)
        self.word_to_idx = {self.padding_value: 0}  # 默认值映射到索引 0
        self.embedding_layer = None
        self.hash_table = None
        self.second_sep_item_num = second_sep_item_num
        self.pooling = pooling

    def build(self, input_shape: tf.TensorShape):
        # 初始化哈希表
        self.split_transform = SplitTransform(
            first_sep=self.first_sep,
            second_sep=self.second_sep,
            second_sep_pos=self.second_sep_pos,
            padding_value=self.padding_value,
            second_sep_item_num=self.second_sep_item_num,
            max_length=self.max_length,
        )
        self.varlen_emb = StrEmbedding(
            embedding_dim=self.embedding_dim,
            vocab_size=self.vocab_size,
            padding_value=self.padding_value,
            pooling=self.pooling,
        )
        super(SplitEmbedding, self).build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        extracted_data = self.split_transform(inputs)
        embedded_data = self.varlen_emb(extracted_data)

        return embedded_data

    def get_config(self):
        config = super(SplitEmbedding, self).get_config()
        config.update(
            {
                "first_sep": self.first_sep,
                "second_sep": self.second_sep,
                "second_sep_pos": self.second_sep_pos,
                "padding_value": self.padding_value,
                "max_length": self.max_length,
                "embedding_dim": self.embedding_dim,
                "vocab_size": self.vocab_size,
                "second_sep_item_num": self.second_sep_item_num,
                "pooling": self.pooling,
            }
        )
        return config


if __name__ == "__main__":
    from tensorflow.keras.layers import (
        Dense,
        InputLayer,
        Layer,
    )
    from tensorflow.keras.models import Sequential

    print("testing SplitEmbedding")

    # 构建模型
    def build_model():
        model = Sequential(
            [
                InputLayer(
                    input_shape=(), dtype=tf.string
                ),  # 使用 InputLayer 定义输入层
                SplitEmbedding(
                    first_sep="|",
                    second_sep="#",
                    second_sep_pos=0,
                    padding_value="PAD",
                    max_length=5,
                    embedding_dim=10,
                    vocab_size=1000,
                    pooling="mean",
                ),
                Dense(1, activation="sigmoid"),  # 二分类输出层
            ]
        )
        return model

    # 测试数据
    input_data = tf.constant(
        [
            "德邦证券,吕品#5|中信建投证券,卢昊#2",
            "国海证券,刘熹#2|东北证券,王凤华#1|中航证券,李蔚#1",
        ]
    )
    labels = tf.constant([1, 0])  # 假设是二分类标签
    # 构建模型
    model = build_model()
    # 编译模型
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # 打印模型摘要
    model.summary()
    # 训练模型
    model.fit(input_data, labels, epochs=2)
    # 测试预测
    predictions = model.predict(input_data)
    print("预测结果:", predictions)
    print("testing finished")
