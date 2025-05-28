import tensorflow as tf
from tensorflow.keras.layers import Layer


class SumPooling(Layer):
    """sum pooling over one dim"""

    def __init__(self, dim: int = 1, keepdims: bool = False, **kwargs):
        super(SumPooling, self).__init__(**kwargs)
        self.keepdims = keepdims
        self.dim = dim

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.dim, keepdims=self.keepdims)

    def get_config(self):
        config = super(SumPooling, self).get_config()
        config.update({"keepdims": self.keepdims, "dim": self.dim})
        return config


if __name__ == "__main__":
    from tensorflow.keras.layers import (
        Dense,
        InputLayer,
        Layer,
    )
    from tensorflow.keras.models import Sequential

    from .as_type import AsType
    from .split_transform import SplitTransform

    print("testing SumPooling")

    # 测试数据
    input_data = tf.constant(
        [
            "德邦证券,吕品#5|中信建投证券,卢昊#2",
            "国海证券,刘熹#2|东北证券,王凤华#1|中航证券,李蔚#1",
        ]
    )

    func = SplitTransform(
        first_sep="|",
        second_sep="#",
        second_sep_pos=1,
        padding_value="0",
        max_length=5,
    )

    f = func(input_data)
    print("拆分后:", f)
    y = AsType(tf.float32)(f)
    print("y:", y)

    s = SumPooling()(y)
    print("求和后:", s)

    s = SumPooling()(y)
    print("求和后:", s)

    def build_model():
        model = Sequential(
            [
                InputLayer(
                    input_shape=(), dtype=tf.string
                ),  # 使用 InputLayer 定义输入层
                SplitTransform(
                    first_sep="|",
                    second_sep="#",
                    second_sep_pos=1,
                    padding_value="0",
                    max_length=5,
                ),
                AsType(tf.float32),
                SumPooling(keepdims=True),
                Dense(1, activation="sigmoid"),  # 二分类输出层
            ]
        )
        return model

    labels = tf.constant([1, 0])  # 假设是二分类标签
    # 构建模型
    model = build_model()
    # 编译模型
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # 打印模型摘要
    model.summary()
    # 训练模型
    model.fit(input_data, labels, epochs=2)
    # 测试预测
    predictions = model.predict(input_data)
    print("预测结果:", predictions)
    print("testing finished")
