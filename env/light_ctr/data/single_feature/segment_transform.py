import tensorflow as tf
from tensorflow.keras.layers import Layer


class SegmentTransform(Layer):
    """
    自定义 Keras 层，用于将连续特征根据分桶离散化。

    参数:
    - segment_list: 分桶的边界值列表。
    - value_dtype: 输出数据类型，默认为 tf.int32。
    """

    def __init__(self, segment_list, value_dtype=tf.int32, **kwargs):
        super(SegmentTransform, self).__init__(**kwargs)
        self.segment_list = segment_list
        self.value_dtype = value_dtype

    def call(self, inputs):
        """
        将输入特征根据分桶离散化。

        参数:
        - inputs: 输入特征张量，形状为 (batch_size, 1)。

        返回:
        - 离散化后的特征张量，形状为 (batch_size, 1)。
        """
        # 初始化结果为最大分桶值 + 1
        result = tf.ones_like(inputs, dtype=self.value_dtype) * (
            len(self.segment_list) + 1
        )

        # 遍历分桶边界值
        for i, segment_value in enumerate(self.segment_list):
            # 判断输入值是否小于当前分桶边界值
            mask = tf.less(inputs, segment_value)
            # 更新结果
            result = tf.where(
                mask, tf.ones_like(result, dtype=self.value_dtype) * (i + 1), result
            )

        return result

    def get_config(self):
        """
        获取层的配置，用于序列化。
        """
        config = super(SegmentTransform, self).get_config()
        config.update(
            {
                "segment_list": self.segment_list,
                "value_dtype": self.value_dtype,
            }
        )
        return config


if __name__ == "__main__":
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, InputLayer
    from tensorflow.keras.models import Sequential

    # 测试数据
    input_data = tf.constant([[0.5], [1.5], [2.5], [3.5], [4.5]], dtype=tf.float32)
    labels = tf.constant([0, 1, 2, 3, 4], dtype=tf.int32)  # 假设是多分类标签

    # 构建模型
    def build_model():
        model = Sequential(
            [
                InputLayer(
                    input_shape=(1,), dtype=tf.float32
                ),  # 使用 InputLayer 定义输入层
                SegmentTransform(
                    segment_list=[1.0, 2.0, 3.0, 4.0], value_dtype=tf.int32
                ),  # 自定义 SegmentTransform
                Dense(5, activation="softmax"),  # 多分类输出层
            ]
        )
        return model

    # 构建模型
    model = build_model()

    # 编译模型
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # 打印模型摘要
    model.summary()

    # 训练模型
    model.fit(input_data, labels, epochs=2)

    # 测试预测
    predictions = model.predict(input_data)
    print("预测结果:", predictions)
    print("testing finished")
