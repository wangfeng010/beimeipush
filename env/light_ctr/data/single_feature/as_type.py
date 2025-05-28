import tensorflow as tf


class AsType(tf.keras.layers.Layer):
    def __init__(self, target_dtype=tf.float32, **kwargs):
        """
        初始化数据类型转换层。
        :param target_dtype: 目标数据类型，默认为 tf.float32。
        :param kwargs: 其他参数。
        """
        super(AsType, self).__init__(**kwargs)
        self.target_dtype = target_dtype

    def call(self, inputs):
        """
        将输入张量转换为目标数据类型。
        :param inputs: 输入张量，可以是任意形状。
        :return: 转换后的张量，数据类型为 target_dtype。
        """
        if self.target_dtype == tf.string:
            # 使用 tf.as_string 将数值类型转换为字符串
            return tf.as_string(inputs)
        elif inputs.dtype == tf.string and self.target_dtype in (tf.int32, tf.float32):
            # 使用 tf.strings.to_number 将字符串转换为数值类型
            return tf.strings.to_number(inputs, out_type=self.target_dtype)
        else:
            # 使用 tf.cast 进行其他类型转换
            return tf.cast(inputs, dtype=self.target_dtype)

    def get_config(self):
        # 支持序列化
        config = super(AsType, self).get_config()
        config.update(
            {"target_dtype": self.target_dtype}
        )  # 保存目标数据类型的名称
        return config


if __name__ == "__main__":
    # 测试 1: int32 转 float32
    input_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
    conversion_layer = AsType(target_dtype=tf.float32)
    output_data = conversion_layer(input_data)
    print("int32 转 float32:")
    print(output_data.numpy())

    # 测试 2: float32 转 int32
    input_data = tf.constant([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=tf.float32)
    conversion_layer = AsType(target_dtype=tf.int32)
    output_data = conversion_layer(input_data)
    print("\nfloat32 转 int32:")
    print(output_data.numpy())

    # 测试 3: float32 转 string
    input_data = tf.constant([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=tf.float32)
    conversion_layer = AsType(target_dtype=tf.string)
    output_data = conversion_layer(input_data)
    print("\nfloat32 转 string:")
    print(output_data.numpy())

    # 测试 4: string 转 int32
    input_data = tf.constant([["1", "2", "3"], ["4", "5", "6"]], dtype=tf.string)
    conversion_layer = AsType(target_dtype=tf.int32)
    output_data = conversion_layer(input_data)
    print("\nstring 转 int32:")
    print(output_data.numpy())

    # 测试 5: int32 转 string
    input_data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
    conversion_layer = AsType(target_dtype=tf.string)
    output_data = conversion_layer(input_data)
    print("\nint32 转 string:")
    print(output_data.numpy())
