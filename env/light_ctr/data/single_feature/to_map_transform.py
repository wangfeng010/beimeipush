import tensorflow as tf
from typing import Dict, Optional, Any, Tuple
from tensorflow.keras.layers import Dense, Embedding, Input, Layer
from tensorflow.keras.models import Model


class ToMapTransform(Layer):
    """
    将字符串转换为键值映射的层。

    该层将输入字符串按照两级分隔符进行拆分，提取键和值，生成对应的映射关系。
    支持多种映射类型，如分数映射、排名映射和分类映射。

    Attributes:
        config: 包含以下配置项的字典:
            first_sep (str): 第一次拆分的分隔符，如","
            second_sep (str): 第二次拆分的分隔符，如":"
            key_position (int): 提取key的位置索引
            value_position (int, optional): 提取value的位置索引，对于rank_map可为None
            map_type (str): 映射类型，支持"score_map"、"rank_map"、"class_map"
            max_length (int): 每个样本最多保留的元素数量
            first_pad_value (str): 第一次拆分后的填充值
            second_pad_value (str): 第二次拆分后的填充值（必须是合法的数字字符串）
    """

    # 定义默认配置
    DEFAULT_CONFIG = {
        "first_sep": ",",
        "second_sep": ":",
        "key_position": 0,
        "value_position": 1,
        "map_type": "score_map",
        "max_length": 100,
        "first_pad_value": "",
        "second_pad_value": "0",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        初始化ToMapTransform层。

        Args:
            config: 配置参数字典，包含分隔符、位置索引等信息
            **kwargs: 传递给父类的其他参数
        """
        super(ToMapTransform, self).__init__(**kwargs)
        
        # 使用默认配置，然后用提供的配置覆盖
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        # 为了向后兼容和方便访问，将配置项作为类属性
        self.first_sep = self.config["first_sep"]  # 第一次拆分的分隔符
        self.second_sep = self.config["second_sep"]  # 第二次拆分的分隔符
        self.key_position = self.config["key_position"]  # 提取key的位置
        self.value_position = self.config["value_position"]  # 提取value的位置
        self.map_type = self.config["map_type"]  # 映射类型
        self.max_length = self.config["max_length"]  # 最大长度
        self.first_pad_value = self.config["first_pad_value"]  # 第一级填充值
        self.second_pad_value = self.config["second_pad_value"]  # 第二级填充值

    def call(self, inputs):
        """
        执行层的前向传播。

        Args:
            inputs: 输入张量，形状为 [batch_size]，数据类型为字符串

        Returns:
            tuple: (keys, values) 两个张量，形状均为 [batch_size, max_length]
        """
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
            split_elements = split_elements.to_tensor(default_value=self.second_pad_value)

            # 提取 key
            keys = split_elements[:, self.key_position]

            # 根据 map_type 提取或生成 values
            if self.map_type in ("score_map", "class_map"):
                # 提取 value 并转换为浮点数
                values = tf.strings.to_number(
                    split_elements[:, self.value_position], 
                    out_type=tf.float32
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
                default_value=0.0, 
                shape=[tf.shape(inputs)[0], self.max_length]
            )

        return keys, values

    def compute_output_shape(self, input_shape):
        """
        计算输出张量的形状。

        Args:
            input_shape: 输入张量的形状

        Returns:
            tuple: 包含两个输出张量形状的元组
        """
        # 返回 keys 和 values 的形状 [batch_size, max_len]
        return (
            (input_shape[0], self.max_length),
            (input_shape[0], self.max_length),
        )

    def get_config(self):
        """
        获取层的配置，用于序列化。

        Returns:
            dict: 包含层配置的字典
        """
        base_config = super().get_config()
        base_config.update({"config": self.config})
        return base_config


if __name__ == "__main__":
    from tensorflow.keras.layers import GlobalAveragePooling1D

    print("测试 ToMapTransform 层...")

    # 测试数据
    input_data = tf.constant(["a:1,b:2,c:3", "d:4,e:5,f:6,g:7"])

    # 创建自定义层实例
    rank_map_config = {
        "first_sep": ",",
        "second_sep": ":",
        "key_position": 0,
        "value_position": 1,
        "map_type": "rank_map",
        "max_length": 5,
        "first_pad_value": "PAD",
        "second_pad_value": "0",
    }
    
    to_map_layer = ToMapTransform(config=rank_map_config)

    # 调用层进行转换
    keys, values = to_map_layer(input_data)

    # 打印结果
    print("Keys:")
    print(keys.numpy())
    print("\nValues:")
    print(values.numpy())

    print("\n构建并测试完整模型...")
    
    # 定义模型输入
    input_layer = Input(shape=(), dtype=tf.string)
    
    # 配置映射转换层
    score_map_config = {
        "first_sep": ",",
        "second_sep": ":",
        "key_position": 0,
        "value_position": 1,
        "map_type": "score_map",
        "max_length": 5,
        "first_pad_value": "PAD",
        "second_pad_value": "0",
    }
    
    # 创建转换层并应用
    to_map_layer = ToMapTransform(config=score_map_config)
    keys, values = to_map_layer(input_layer)

    # 将 keys 转换为整数哈希
    keys_hashed = tf.strings.to_hash_bucket_fast(
        keys, 
        num_buckets=100
    )

    # 添加嵌入层
    embedding_layer = Embedding(
        input_dim=100, 
        output_dim=10
    )(keys_hashed)

    # 使用全局平均池化处理嵌入向量
    pooled_layer = GlobalAveragePooling1D()(embedding_layer)

    # 添加输出层
    output_layer = Dense(1, activation="sigmoid")(pooled_layer)

    # 创建并编译模型
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # 打印模型摘要
    model.summary()
    print("测试完成")
