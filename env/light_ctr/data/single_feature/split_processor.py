import tensorflow as tf
from tensorflow.keras.layers import Layer


class SplitProcessor(Layer):
    """
    简化版本的字符串预处理层，仅支持一次字符串拆分。

    参数:
        sep (str): 分隔符，默认为 "|"。
        padding_value (str): 填充值，默认为 "PAD"。
        max_length (int): 最大长度，默认为 5。

    """

    def __init__(
        self,
        sep="|",
        padding_value="PAD",
        max_length=5,
        **kwargs,
    ):
        # 调用父类的初始化方法，仅传递必要的参数
        super(SplitProcessor, self).__init__(**kwargs)
        self.sep = sep  # 保存分隔符
        self.padding_value = padding_value  # 保存填充值
        self.max_length = max_length  # 保存最大长度

    def call(self, inputs):
        # 1. 按照 sep 拆分成列表
        split_data = tf.strings.split(inputs, self.sep)
        # 将 RaggedTensor 转换为 Tensor，并用 padding_value 填充
        return split_data.to_tensor(
            default_value=self.padding_value, shape=[None, self.max_length]
        )

    def get_config(self):
        # 获取父类的配置，并添加简化版本的参数
        config = super(SplitProcessor, self).get_config()
        config.update(
            {
                "sep": self.sep,
                "padding_value": self.padding_value,
                "max_length": self.max_length,
            }
        )
        return config


if __name__ == "__main__":
    from tensorflow.keras.layers import (
        Dense,
        InputLayer,
    )
    from tensorflow.keras.models import Sequential

    from .str_embedding import StrEmbedding

    print("testing SplitProcessor")

    def build_model():
        model = Sequential(
            [
                InputLayer(
                    input_shape=(), dtype=tf.string
                ),  # 使用 InputLayer 定义输入层
                SplitProcessor(
                    sep="|",
                    padding_value="PAD",
                    max_length=5,
                ),
                StrEmbedding(
                    embedding_dim=10,
                    vocab_size=1000,
                    padding_value="PAD",
                    pooling="mean",
                ),
                Dense(1, activation="sigmoid"),  # 二分类输出层
            ]
        )
        return model

    # 测试数据
    input_data = tf.constant(["德邦证券|中信建投证券", "国海证券|东北证券|中航证券"])
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
