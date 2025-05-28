import tensorflow as tf


class StringConcat(tf.keras.layers.Layer):
    """
    自定义层，用于将两个字符串张量连接起来。

    :param sep: 连接字符串之间的分隔符，默认为空字符串。
    :param kwargs: 其他传递给父类的参数。
    """

    def __init__(self, sep="", **kwargs):
        super(StringConcat, self).__init__(**kwargs)
        self.sep = sep

    def call(self, inputs):
        """
        调用此层以连接两个字符串张量。

        :param inputs: 包含两个字符串张量的列表或元组。
        :return: 连接后的字符串张量。
        """
        str1, str2 = inputs
        if len(str1.shape) > len(str2.shape):
            str2 = tf.expand_dims(str2, axis=-1)
            # 如果 str1 的形状比 str2 大，扩展 str2 的最后一个维度
        elif len(str2.shape) > len(str1.shape):
            str1 = tf.expand_dims(str1, axis=-1)
        return tf.strings.join(
            [str1, str2], separator=self.sep
        )  # 使用指定的分隔符连接两个字符串张量

    def get_config(self):
        """
        获取此层的配置信息，以便于序列化。

        :return: 配置字典。
        """
        config = super(StringConcat, self).get_config()
        config.update({"sep": self.sep})
        return config


if __name__ == "__main__":
    from tensorflow.keras.layers import Dense

    from ..single_feature.str_embedding import StrEmbedding

    # 测试用例
    string_tensor1 = tf.constant(["hello", "world", "foo"], dtype=tf.string)
    string_tensor2 = tf.constant(["TensorFlow", "rocks", "bar"], dtype=tf.string)
    sep = "-"
    concat_layer = StringConcat(sep=sep)
    result = concat_layer((string_tensor1, string_tensor2))
    print(result)

    string_tensor3 = tf.constant([["hello", "world"], ["foo", "test"]], dtype=tf.string)
    string_tensor4 = tf.constant(
        [["TensorFlow", "rocks"], ["bar", "ok"]], dtype=tf.string
    )
    result2 = concat_layer((string_tensor3, string_tensor4))
    print(result2)

    string_tensor5 = tf.constant("hello", dtype=tf.string)
    string_tensor6 = tf.constant("world", dtype=tf.string)
    result3 = concat_layer((string_tensor5, string_tensor6))
    print(result3)

    input1 = tf.keras.Input(shape=(), dtype=tf.string)  # 标量输入
    input2 = tf.keras.Input(shape=(), dtype=tf.string)  # 标量输入
    concat_layer = StringConcat(sep="-")
    s = concat_layer((input1, input2))
    f = StrEmbedding(
        embedding_dim=10,
        vocab_size=1000,
        padding_value="PAD",
        pooling=None,
    )(s)
    y = Dense(1, activation="sigmoid")(f)  # 二分类输出层

    model = tf.keras.Model(inputs=[input1, input2], outputs=y)

    # 编译模型
    model.compile(
        optimizer="adam", loss="mse"
    )  # 这里使用 'mse' 只是为了演示编译，实际应用中需要根据任务选择合适的损失函数

    # 打印模型结构
    model.summary()

    # 创建虚拟数据进行训练测试
    train_data1 = tf.constant(["hello", "world", "foo"], dtype=tf.string)
    train_data2 = tf.constant(["TensorFlow", "rocks", "bar"], dtype=tf.string)
    train_labels = tf.constant([0.0, 0.0, 0.0])  # 虚拟标签，因为损失函数是mse

    model.fit((train_data1, train_data2), train_labels, epochs=3)
    print("test finished.")
