import tensorflow as tf
from tensorflow.keras.layers import Layer


class ClipLayer(Layer):
    """
    对输入特征进行裁剪，将元素值限制在一定范围内。
    """

    def __init__(self, min_value, max_value, **kwargs):
        super(ClipLayer, self).__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        # 使用 tf.clip_by_value 进行裁剪
        clipped_output = tf.clip_by_value(inputs, self.min_value, self.max_value)
        return clipped_output

    def get_config(self):
        config = super(ClipLayer, self).get_config()
        config.update(
            {
                "min_value": self.min_value,
                "max_value": self.max_value,
            }
        )
        return config


if __name__ == "__main__":
    # 测试 ClipLayer 是否能够编译和训练
    # 创建一个简单的模型
    input_layer = tf.keras.Input(shape=(10,))
    clip_layer = ClipLayer(min_value=-1.0, max_value=1.0)(input_layer)
    output_layer = tf.keras.layers.Dense(1)(clip_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # 编译模型
    model.compile(optimizer="adam", loss="mse")

    # 生成一些随机数据
    import numpy as np

    x_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randn(100, 1).astype(np.float32)

    # 训练模型
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    print("testing finished")
