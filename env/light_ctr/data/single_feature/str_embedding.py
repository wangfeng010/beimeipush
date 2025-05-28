from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import (
    Embedding,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Layer,
)


class StrEmbedding(Layer):
    """
    字符串的数值映射与嵌入向量生成，支持字符串与字符串列表。
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        padding_value: str = "PAD",
        pooling: Optional[str] = "mean",  # None, "mean" 或 "max"
        **kwargs,
    ):
        """
        初始化字符串预处理层。

        参数:
            embedding_dim (int): 嵌入向量的维度，默认为 10。
            vocab_size (int): 词汇表大小，默认为 1000。
        """
        super(StrEmbedding, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size  # 固定大小的词表
        self.padding_value = padding_value

        self.vocab = set()
        self.vocab.add(self.padding_value)
        self.word_to_idx = {self.padding_value: 0}  # 默认值映射到索引 0
        self.embedding_layer = None
        self.hash_table = None
        self.pooling = pooling
        self.pool_layer = None

    def build(self, input_shape):
        # 初始化嵌入层
        self.embedding_layer = Embedding(
            input_dim=self.vocab_size, output_dim=self.embedding_dim
        )
        if self.pooling == "mean":
            self.pool_layer = GlobalAveragePooling1D()
        elif self.pooling == "max":
            self.pool_layer = GlobalMaxPooling1D()

        super(StrEmbedding, self).build(input_shape)

    def call(self, inputs):
        # 3. 将字符串映射为索引
        indexed_data = tf.strings.to_hash_bucket_fast(inputs, self.vocab_size)
        # print(indexed_data)
        # 4. 使用嵌入层获取嵌入向量
        embedded_data = self.embedding_layer(indexed_data)
        if self.pool_layer:
            embedded_data = self.pool_layer(embedded_data)

        return embedded_data

    def get_config(self):
        config = super(StrEmbedding, self).get_config()
        config.update(
            {
                "padding_value": self.padding_value,
                "embedding_dim": self.embedding_dim,
                "vocab_size": self.vocab_size,
                "pooling": self.pooling,
            }
        )
        return config


if __name__ == "__main__":
    from tensorflow.keras.layers import (
        Dense,
        GlobalAveragePooling1D,
        InputLayer,
        Layer,
    )
    from tensorflow.keras.models import Sequential

    # 下方是字符串的embedding, 字符串列表见SplitTransform的测试代码
    print("testing SplitTransform")

    # 构建模型
    def build_model():
        model = Sequential(
            [
                InputLayer(
                    input_shape=(), dtype=tf.string
                ),  # 使用 InputLayer 定义输入层
                StrEmbedding(
                    embedding_dim=10,
                    vocab_size=1000,
                    padding_value="PAD",
                    pooling=None,
                ),
                Dense(1, activation="sigmoid"),  # 二分类输出层
            ]
        )
        return model

    # 测试数据
    input_data = tf.constant(["A", "B", "A", "C"])
    labels = tf.constant([1, 0, 1, 0])  # 假设是二分类标签
    # 构建模型
    model = build_model()
    # 编译模型
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # 打印模型摘要
    model.summary()
    # 训练模型
    model.fit(input_data, labels, epochs=10)
    # 测试预测
    predictions = model.predict(input_data)
    print("预测结果:", predictions)
    print("testing finished")
