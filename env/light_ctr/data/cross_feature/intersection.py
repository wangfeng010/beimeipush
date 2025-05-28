import tensorflow as tf


class Intersection(tf.keras.layers.Layer):
    """
    判断两个字符串向量的交集，判断方式是将两个向量转换为集合，然后取交集。
    """
    def __init__(self, **kwargs):
        """
        初始化判断交集的层。
        :param kwargs: 其他参数。
        """
        super(Intersection, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        """
        判断两个张量的取值是否存在交集。
        :param inputs: 包含两个张量的列表，每个张量的形状为 (batch_size, sequence_length)。
        :return: 返回一个张量，形状为 (batch_size,)，值为 1（存在交集）或 0（不存在交集）。
        """
        # 确保输入是两个张量
        if len(inputs) != 2:
            raise ValueError("输入必须是两个张量。")

        # 将输入转换为张量
        tensor1 = tf.convert_to_tensor(inputs[0], dtype=tf.string)
        tensor2 = tf.convert_to_tensor(inputs[1], dtype=tf.string)

        # 使用 tf.map_fn 对每个样本进行处理
        def has_intersection(t1, t2):
            # 判断两个张量是否存在交集
            intersection = tf.sets.intersection(t1[tf.newaxis], t2[tf.newaxis])
            return tf.cond(
                tf.size(intersection) > 0,
                lambda: tf.constant(1, dtype=tf.int32),
                lambda: tf.constant(0, dtype=tf.int32),
            )

        # 对批量输入进行处理
        batch_result = tf.map_fn(
            lambda x: has_intersection(x[0], x[1]), (tensor1, tensor2), dtype=tf.int32
        )
        return batch_result

    def get_config(self):
        # 支持序列化
        config = super(Intersection, self).get_config()
        return config


if __name__ == "__main__":
    from tensorflow.keras.layers import Dense

    from ..single_feature.as_type import AsType
    from ..single_feature.str_embedding import StrEmbedding

    # 示例：在模型中使用自定义层
    # 定义输入
    input1 = tf.keras.Input(
        shape=(None,), dtype=tf.string, name="input1"
    )  # 形状: (batch_size, sequence_length)
    input2 = tf.keras.Input(
        shape=(None,), dtype=tf.string, name="input2"
    )  # 形状: (batch_size, sequence_length)

    # 使用自定义层判断交集
    intersection_output = Intersection()([input1, input2])
    x = AsType(target_dtype=tf.string)(intersection_output)
    f = StrEmbedding(
        embedding_dim=10,
        vocab_size=1000,
        padding_value="PAD",
        pooling=None,
    )(x)
    y = Dense(1, activation="sigmoid")(f)  # 二分类输出层

    # 构建模型
    model = tf.keras.Model(inputs=[input1, input2], outputs=y)

    # 测试模型
    str1 = tf.constant(
        [["apple", "banana", "cherry"], ["dog", "cat", "fish"]], dtype=tf.string
    )  # 批量输入
    str2 = tf.constant(
        [["banana", "grape", "kiwi"], ["bird", "fish", "lion"]], dtype=tf.string
    )  # 批量输入
    result = model([str1, str2])

    print("判断结果:")
    print(result.numpy())  # 输出: [1, 1]（存在交集）

    # 测试编译和训练
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 模拟训练数据
    y_true = tf.constant([1, 1], dtype=tf.int32)  # 假设两个样本都存在交集
    model.fit([str1, str2], y_true, epochs=2)
    print("训练完成.")
    print("测试用例完成.")
