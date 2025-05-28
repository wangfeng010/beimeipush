import tensorflow as tf
from tensorflow.keras.layers import Layer


class Log1pTransform(Layer):
    """
    对输入特征取 log(x + 1) 变换。
    """

    def __init__(self, **kwargs):
        super(Log1pTransform, self).__init__(**kwargs)

    def call(self, inputs):
        # 使用 tf.math.log1p 进行 log(x + 1) 变换
        log_transformed = tf.math.log1p(inputs)
        return log_transformed

    def get_config(self):
        config = super(Log1pTransform, self).get_config()
        return config


if __name__ == "__main__":
    # 测试 Log1pTransform 是否能够编译和训练
    # 创建一个简单的模型
    input_layer = tf.keras.Input(shape=(10,))
    log1p_transform_layer = Log1pTransform()(input_layer)
    output_layer = tf.keras.layers.Dense(1)(log1p_transform_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # 编译模型
    model.compile(optimizer="adam", loss="mse")

    # 生成一些随机数据
    import numpy as np

    x_train = np.abs(np.random.randn(100, 10).astype(np.float32))  # 确保输入为非负数
    y_train = np.random.randn(100, 1).astype(np.float32)

    # 训练模型
    model.fit(x_train, y_train, epochs=10, batch_size=32)
