import tensorflow as tf
from tensorflow.keras.layers import Layer


class FillNaString(Layer):
    """
    对输入特征进行空字符串或 "NaN" 的填充。
    """

    def __init__(self, fill_value="0", **kwargs):
        super(FillNaString, self).__init__(**kwargs)
        self.fill_value = fill_value  # 用于填充空字符串的值

    def call(self, inputs):
        # 使用 tf.where 和 tf.equal 来替换空字符串和 "NaN"
        condition = tf.logical_or(
            tf.equal(inputs, ""),  # 检测空字符串
            tf.logical_or(
                tf.equal(inputs, "NaN"),  # 检测 "NaN"
                tf.equal(inputs, "NULL"),  # 检测 "NULL"
            ),
        )
        # 将条件为 True 的位置替换为 fill_value
        filled_output = tf.where(condition, self.fill_value, inputs)
        return filled_output

    def get_config(self):
        config = super(FillNaString, self).get_config()
        config.update(
            {
                "fill_value": self.fill_value,
            }
        )
        return config


if __name__ == "__main__":
    # 测试 FillNaString 是否能够编译和训练
    # 创建一个简单的模型
    input_layer = tf.keras.Input(shape=(10,), dtype=tf.string)
    fill_na_string_layer = FillNaString(fill_value="0")(input_layer)

    # 将字符串转换为浮点数以便后续处理
    to_float_layer = tf.keras.layers.Lambda(
        lambda x: tf.strings.to_number(x, out_type=tf.float32)
    )(fill_na_string_layer)

    # 确保输出的形状是 (batch_size, 10)
    to_float_layer = tf.ensure_shape(to_float_layer, [None, 10])

    # 添加 Dense 层
    output_layer = tf.keras.layers.Dense(1)(to_float_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # 编译模型
    model.compile(optimizer="adam", loss="mse")

    # 生成一些随机数据，并插入空字符串和 "NaN"
    import numpy as np

    x_train = np.array(
        [
            ["1.0", "2.0", "", "NaN", "3.0", "4.0", "", "5.0", "NaN", "6.0"]
            for _ in range(100)
        ],
        dtype=np.str_,
    )
    y_train = np.random.randn(100, 1).astype(np.float32)

    # 训练模型
    model.fit(x_train, y_train, epochs=10, batch_size=32)
