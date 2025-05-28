import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Input, Layer
from tensorflow.keras.models import Model


class ToMapTransform(Layer):
    def __init__(
        self,
        first_sep,
        second_sep,
        key_position,
        value_position=None,
        map_type="score_map",
        max_length=100,
        first_pad_value="",
        second_pad_value="0",
        **kwargs,
    ):
        super(ToMapTransform, self).__init__(**kwargs)
        self.first_sep = first_sep  # 第一次拆分的分隔符
        self.second_sep = second_sep  # 第二次拆分的分隔符
        self.key_position = key_position  # 提取 key 的位置
        self.value_position = value_position  # 提取 value 的位置
        self.map_type = map_type  # 映射类型：score_map, rank_map, class_map
        self.max_length = max_length  # 每个样本最多保留的元素数量
        self.first_pad_value = first_pad_value  # 第一次拆分后的填充值
        self.second_pad_value = (
            second_pad_value  # 第二次拆分后的填充值（必须是合法的数字字符串）
        )

    def call(self, inputs):
        # 使用 tf.strings.split 按第一个分隔符分割字符串
        split_inputs = tf.strings.split(inputs, sep=self.first_sep)

        # 对第一次拆分的结果进行填充
        split_inputs = split_inputs.to_tensor(
            default_value=self.first_pad_value,
            shape=[tf.shape(inputs)[0], self.max_length],
        )

        # 定义处理每个样本的函数
        def process_sample(sample):
            # 使用 tf.strings.split 按第二个分隔符分割每个元素
            split_elements = tf.strings.split(sample, sep=self.second_sep)

            # 将 RaggedTensor 转换为普通张量并进行填充
            split_elements = split_elements.to_tensor(
                default_value=self.second_pad_value
            )

            # 提取 key
            keys = split_elements[:, self.key_position]

            # 根据 map_type 提取或生成 values
            if self.map_type == "score_map" or self.map_type == "class_map":
                # 提取 value 并转换为浮点数
                values = tf.strings.to_number(
                    split_elements[:, self.value_position], out_type=tf.float32
                )
            elif self.map_type == "rank_map":
                # 生成排名值（从 1 开始）
                values = tf.range(tf.shape(split_elements)[0], dtype=tf.float32) + 1.0
            else:
                raise ValueError(
                    "Invalid map_type. Supported types are 'score_map', 'rank_map', and 'class_map'."
                )

            return keys, values

        # 使用 tf.map_fn 对每个样本进行处理
        keys, values = tf.map_fn(
            process_sample,
            split_inputs,
            fn_output_signature=(
                tf.TensorSpec(shape=[None], dtype=tf.string),
                tf.TensorSpec(shape=[None], dtype=tf.float32),
            ),
        )

        # 对 keys 和 values 进行填充
        if isinstance(keys, tf.RaggedTensor):
            keys = keys.to_tensor(
                default_value=self.second_pad_value,
                shape=[tf.shape(inputs)[0], self.max_length],
            )
        if isinstance(values, tf.RaggedTensor):
            values = values.to_tensor(
                default_value=0.0, shape=[tf.shape(inputs)[0], self.max_length]
            )

        return keys, values

    def compute_output_shape(self, input_shape):
        # 返回 keys 和 values 的形状 [batch_size, max_len]
        return [
            (input_shape[0], self.max_length),
            (input_shape[0], self.max_length),
        ]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "first_sep": self.first_sep,
                "second_sep": self.second_sep,
                "key_position": self.key_position,
                "map_type": self.map_type,
                "max_length": self.max_length,
                "first_pad_value": self.first_pad_value,
                "second_pad_value": self.second_pad_value,
            }
        )


if __name__ == "__main__":
    from tensorflow.keras.layers import GlobalAveragePooling1D

    # 测试数据
    input_data = tf.constant(["a:1,b:2,c:3", "d:4,e:5,f:6,g:7"])

    # 创建自定义层
    to_map_layer = ToMapTransform(
        first_sep=",",
        second_sep=":",
        key_position=0,
        value_position=1,
        map_type="rank_map",  # 映射类型
        max_length=5,  # 每个样本最多保留 5 个元素
        first_pad_value="PAD",  # 第一次拆分后的填充值
        second_pad_value="0",  # 第二次拆分后的填充值（必须是合法的数字字符串）
    )

    # 调用层
    keys, values = to_map_layer(input_data)

    # 打印结果
    print("Keys:")
    print(keys.numpy())

    print("Values:")
    print(values.numpy())

    # 定义模型
    input_data = Input(shape=(), dtype=tf.string)  # 输入是字符串
    to_map_layer = ToMapTransform(
        first_sep=",",
        second_sep=":",
        key_position=0,
        value_position=1,
        map_type="score_map",  # 映射类型
        max_length=5,  # 每个样本最多保留 5 个元素
        first_pad_value="PAD",  # 第一次拆分后的填充值
        second_pad_value="0",  # 第二次拆分后的填充值（必须是合法的数字字符串）
    )
    keys, value = to_map_layer(input_data)

    # 将 keys 转换为整数哈希
    keys_hashed = tf.strings.to_hash_bucket_fast(
        keys, num_buckets=100
    )  # 假设哈希桶数量为 100

    # 添加嵌入层
    embedding_layer = Embedding(input_dim=100, output_dim=10)(
        keys_hashed
    )  # 嵌入维度为 10

    # 将嵌入结果展平（只展平 max_len 和 embedding_dim，保留 batch 维度）
    flatten_layer = GlobalAveragePooling1D()(
        embedding_layer
    )  # 对嵌入向量进行全局平均池化
    # 形状为 [batch_size, max_len * emb_dim]

    # 添加全连接层
    output = Dense(1, activation="sigmoid")(flatten_layer)  # 假设是二分类任务

    # 创建模型
    model = Model(inputs=input_data, outputs=output)

    # 编译模型
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",  # 假设是二分类任务
        metrics=["accuracy"],
    )

    # 打印模型摘要
    model.summary()
    print("testing finished")
