from typing import List

from tensorflow.keras import layers


class DNN(layers.Layer):
    """
    DNN 层，用于 DIN 等深度模型。

    参数:
        - **dnn_units** (List[int]) - 每个全连接层的单元数。
        - **activation** (str) - 全连接层的激活函数。
    """

    def __init__(self, dnn_units: List[int], activation="relu", **kwargs):
        super(DNN, self).__init__()
        self.dnn_units = dnn_units
        self.activation = activation

    def build(self, input_shape):
        self.fc_layers = []
        for i, units in enumerate(self.dnn_units):
            self.fc_layers.append(layers.Dense(units, activation=self.activation))
        super(DNN, self).build(input_shape)

    def call(self, inputs):
        """
        前向传播方法。

        :param inputs: 输入张量。
        :param training: 是否在训练模式下，用于控制 dropout 层的行为。
        :return: 输出张量。
        """
        x = inputs
        for layer in self.fc_layers:
            x = layer(x)
        return x

    def compute_output_shape(self, input_shape):
        """
        计算输出张量的形状。

        :param input_shape: 输入张量的形状。
        :return: 输出张量的形状。
        """
        # 最后一层的输出形状就是 DNN 的输出形状
        output_shape = input_shape
        for layer in self.fc_layers:
            output_shape = layer.compute_output_shape(output_shape)
        return output_shape

    def get_config(self):
        """
        返回层的配置信息。

        :return: 层的配置信息字典。
        """
        config = super(DNN, self).get_config()
        config.update(
            {
                "dnn_units": self.dnn_units,
                "activation": self.activation,
            }
        )
        return config


if __name__ == "__main__":
    # 创建一个完整的 Keras 模型
    import numpy as np
    from tensorflow.keras import Model, layers

    def create_model(input_dim, dnn_units):
        inputs = layers.Input(shape=(input_dim,))
        dnn_output = DNN(dnn_units)(inputs)
        outputs = layers.Dense(1, activation="sigmoid")(dnn_output)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    # 测试模型
    input_dim = 10
    dnn_units = [128, 64]

    model = create_model(input_dim, dnn_units)

    # 编译模型
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 生成随机数据
    X_train = np.random.uniform(size=(1000, input_dim))
    y_train = np.random.randint(2, size=(1000, 1))

    # 训练模型
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

    # 生成测试数据
    X_test = np.random.uniform(size=(200, input_dim))
    y_test = np.random.randint(2, size=(200, 1))

    # 评估模型
    results = model.evaluate(X_test, y_test)
    print("Test Loss and Accuracy:", results)
