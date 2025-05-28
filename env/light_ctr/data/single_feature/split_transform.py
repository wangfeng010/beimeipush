import tensorflow as tf
from tensorflow.keras.layers import Layer


class SplitTransform(Layer):
    """
    字符串预处理层，支持两次字符串拆分。
    """

    def __init__(
        self,
        first_sep="|",
        second_sep="#",
        second_sep_pos=0,
        padding_value="PAD",
        second_sep_item_num=2,
        max_length=5,
        **kwargs,
    ):
        """
        初始化字符串预处理层。

        参数:
            first_sep (str): 第一次拆分的分隔符，默认为 "|"。
            second_sep (str): 第二次拆分的分隔符，默认为 "#"。
            second_sep_pos (int): 第二次拆分后提取的下标，默认为 0。
            padding_value (str): 填充值，默认为 "PAD"。
            second_sep_item_num (int): 第二次拆分后提取的元素个数，默认为 2。
            max_length (int): 最大长度，默认为 5。
            vocab_size (int): 词汇表大小，默认为 1000。
        """
        super(SplitTransform, self).__init__(**kwargs)
        self.first_sep = first_sep
        self.second_sep = second_sep
        self.second_sep_pos = second_sep_pos
        self.padding_value = padding_value
        self.max_length = max_length
        self.second_sep_item_num = second_sep_item_num

    def build(self, input_shape):
        super(SplitTransform, self).build(input_shape)

    def call(self, inputs):
        # 1. 按照 first_sep 拆分成列表
        split_data = tf.strings.split(inputs, self.first_sep)
        # 将 RaggedTensor 转换为 Tensor，并用 padding_value 填充
        split_data = split_data.to_tensor(
            default_value=self.padding_value, shape=[None, self.max_length]
        )

        # 2. 对于列表元素进行二次拆分（second_sep="#"），然后提取指定下标（second_sep_pos=0）的元素
        def extract_element(seq):
            # 将每个元素按照 second_sep 拆分
            split_elements = tf.strings.split(seq, self.second_sep)
            # 将 RaggedTensor 转换为 Tensor，并用 padding_value 填充
            split_elements = split_elements.to_tensor(
                default_value=self.padding_value, shape=[None, self.second_sep_item_num]
            )  # 假设最多拆分成两部分
            # 提取指定下标的元素
            return split_elements[:, self.second_sep_pos]

        return tf.map_fn(extract_element, split_data, dtype=tf.string)

    def compute_output_shape(self, input_shape):
        batch = input_shape[0]
        return (batch, self.max_length)

    def get_config(self):
        config = super(SplitTransform, self).get_config()
        config.update(
            {
                "first_sep": self.first_sep,
                "second_sep": self.second_sep,
                "second_sep_pos": self.second_sep_pos,
                "padding_value": self.padding_value,
                "max_length": self.max_length,
                "second_sep_item_num": self.second_sep_item_num,
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

    from .str_embedding import StrEmbedding

    print("testing SplitTransform")

    # 构建模型
    def build_model():
        model = Sequential(
            [
                InputLayer(
                    input_shape=(), dtype=tf.string
                ),  # 使用 InputLayer 定义输入层
                SplitTransform(
                    first_sep="|",
                    second_sep="#",
                    second_sep_pos=0,
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
