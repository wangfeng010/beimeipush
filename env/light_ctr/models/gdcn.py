import tensorflow as tf
from tensorflow.keras import layers


class GatedCrossNetwork(layers.Layer):
    """
    Keras版本的Gated Cross Network。

    参数:
    - dim_in: 输入维度。
    - layer_num: 层数，默认为3。

    输入:
    - Tensor (batch_size, dim_in)

    输出:
    - Tensor (batch_size, dim_in)
    """

    def __init__(self, dim_in: int, layer_num: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.dim_in = dim_in
        self.layer_num = layer_num

    def build(self, input_shape):
        self.W = []
        self.Wg = []
        self.b = []
        for _ in range(self.layer_num):
            self.W.append(
                self.add_weight(
                    shape=(self.dim_in, self.dim_in),
                    initializer="glorot_uniform",  # 或 'uniform'
                    trainable=True,
                    name="W",
                )
            )
            self.Wg.append(
                self.add_weight(
                    shape=(self.dim_in, self.dim_in),
                    initializer="glorot_uniform",  # 或 'uniform'
                    trainable=True,
                    name="Wg",
                )
            )
            self.b.append(
                self.add_weight(
                    shape=(self.dim_in,),
                    initializer="uniform",
                    trainable=True,
                    name="b",
                )
            )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """前向传播。

        参数:
            x: Tensor (batch_size, dim_in)

        返回:
            Tensor (batch_size, dim_in)
        """
        xi = x
        for i in range(self.layer_num):
            weight = tf.sigmoid(tf.matmul(xi, self.Wg[i]))
            f = tf.matmul(xi, self.W[i]) + self.b[i]
            xi = x * f * weight + xi
        return xi

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim_in": self.dim_in,
                "layer_num": self.layer_num,
            }
        )
        return config


if __name__ == "__main__":
    import numpy as np
    from tensorflow.keras import Model, layers

    # 创建一个完整的 Keras 模型
    def create_model(input_dim, layer_num):
        inputs = layers.Input(shape=(input_dim,))
        gated_cross_network = GatedCrossNetwork(dim_in=input_dim, layer_num=layer_num)(
            inputs
        )
        outputs = layers.Dense(1, activation="sigmoid")(gated_cross_network)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    # 测试模型
    input_dim = 10
    layer_num = 3

    model = create_model(input_dim, layer_num)

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
