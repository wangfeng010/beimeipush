import tensorflow as tf
from tensorflow.keras.layers import Layer


class FMLayer(Layer):
    """
    自定义 FM 层，用于计算二阶交互项。

    :param inputs: 输入张量，形状为 (batch_size, num_fields, embedding_dim)
    :return: 输出张量，形状为 (batch_size, 1)
    """

    def __init__(self, **kwargs):
        """
        初始化 FM 层。

        :param kwargs: 其他传递给父类的参数
        """
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        前向传播逻辑。

        :param inputs: 输入张量，形状为 (batch_size, num_fields, embedding_dim)
        :return: 输出张量，形状为 (batch_size, 1)
        """
        # 计算所有特征向量的和
        sum_of_inputs = tf.reduce_sum(
            inputs, axis=1, keepdims=True
        )  # (batch_size, 1, embedding_dim)

        # 计算所有特征向量的平方和
        square_of_sum = tf.square(sum_of_inputs)  # (batch_size, 1, embedding_dim)

        # 计算每个特征向量的平方
        square_of_inputs = tf.square(inputs)  # (batch_size, num_fields, embedding_dim)

        # 计算所有特征向量平方的和
        sum_of_square = tf.reduce_sum(
            square_of_inputs, axis=1, keepdims=True
        )  # (batch_size, 1, embedding_dim)

        # 计算二阶交互项
        fm_part = 0.5 * tf.reduce_sum(
            square_of_sum - sum_of_square, axis=1, keepdims=True
        )  # (batch_size, 1, 1)

        return fm_part


# 测试自定义 FM 层
if __name__ == "__main__":
    import numpy as np

    # 创建一个输入张量
    inputs = tf.random.uniform(
        (10, 64, 10)
    )  # 形状为 (batch_size, num_fields, embedding_dim)

    # 创建 FM 层
    fm_layer = FMLayer()

    # 调用 FM 层
    outputs = fm_layer(inputs)

    # 打印输出张量的形状
    print(outputs.shape)  # 输出: (10, 1)

    # 创建一个模型并使用 FM 层
    input_layer = tf.keras.Input(shape=(64, 10))  # 形状为 (num_fields, embedding_dim)
    fm_output = fm_layer(input_layer)
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(fm_output)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # 编译模型
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 打印模型结构
    model.summary()

    # 生成虚拟数据
    np.random.seed(42)
    data = np.random.random((1000, 64, 10))  # 形状为 (1000, 64, 10)
    labels = np.random.randint(0, 2, size=(1000, 1))  # 形状为 (1000, 1)

    # 训练模型
    model.fit(data, labels, epochs=10, batch_size=32)
