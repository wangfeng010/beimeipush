from typing import Dict, Optional, Any

import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow_hub as tf_hub
import tensorflow_text as text  # 需要这个依赖处理BERT文本


class BertEmbedding(Layer):
    """
    基于BERT的文本embedding处理类，用于生成文本的向量表示。
    
    使用预训练的BERT模型将文本转换为固定维度的向量表示。
    支持多种BERT模型，如BERT-Base、BERT-Large等。
    
    Attributes:
        config: 包含以下配置项的字典：
            model_url (str): BERT模型的TF-Hub URL地址
            output_dim (int): 输出向量的维度，如果为None，则使用BERT输出的原始维度
            max_seq_length (int): 输入序列的最大长度，默认为128
            pooling_strategy (str): 池化策略，可选值为"cls"（使用[CLS]标记的embedding）、
                                   "mean"（平均池化）或"none"（不池化），默认为"cls"
            trainable (bool): 是否允许BERT模型参数更新，默认为False
            padding_value (str): 填充值，默认为""
    """

    # 默认配置
    DEFAULT_CONFIG = {
        "model_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2",
        "output_dim": None,  # 如果为None，则使用BERT输出的原始维度
        "max_seq_length": 128,
        "pooling_strategy": "cls",  # 'cls', 'mean', 'none'
        "trainable": False,
        "padding_value": "",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        初始化BertEmbedding层。

        Args:
            config: 配置参数字典，包含BERT模型URL、输出维度等信息
            **kwargs: 传递给父类的其他参数
        """
        # 首先调用父类的初始化方法，只传递keras能理解的参数
        super(BertEmbedding, self).__init__(**kwargs)
        
        # 使用默认配置，然后用提供的配置覆盖
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # 为了方便访问，将配置项作为类属性
        self.model_url = self.config["model_url"]
        self.output_dim = self.config["output_dim"]
        self.max_seq_length = self.config["max_seq_length"]
        self.pooling_strategy = self.config["pooling_strategy"]
        self.trainable = self.config["trainable"]
        self.padding_value = self.config["padding_value"]
        
        # 将在build方法中初始化的变量
        self.bert_model = None
        self.preprocessing_model = None
        self.projection = None

    def build(self, input_shape):
        """构建层，初始化BERT模型和预处理层"""
        # 使用BERT预处理文本的模型
        self.preprocessing_model = tf_hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            trainable=False,
            name="bert_preprocessing",
        )
        
        # 加载BERT模型
        self.bert_model = tf_hub.KerasLayer(
            self.model_url,
            trainable=self.trainable,
            name="bert_encoder",
        )
        
        # 如果指定了输出维度且与BERT输出维度不同，则添加投影层
        if self.output_dim is not None:
            self.projection = tf.keras.layers.Dense(
                self.output_dim, activation=None, name="bert_projection"
            )
        
        super(BertEmbedding, self).build(input_shape)

    def call(self, inputs):
        """执行层的前向传播"""
        # 首先处理缺失值
        inputs = tf.strings.regex_replace(
            inputs, tf.constant("^$"), self.padding_value
        )
        
        # 使用BERT预处理文本 - 预处理模型接受原始字符串而不是字典
        preprocessed = self.preprocessing_model(inputs)
        
        # 获取BERT嵌入
        bert_outputs = self.bert_model(preprocessed)
        
        # 根据池化策略选择适当的输出
        if self.pooling_strategy == "cls":
            # 使用[CLS]标记的embedding作为整句表示
            pooled_output = bert_outputs["pooled_output"]  # [batch_size, hidden_size]
        elif self.pooling_strategy == "mean":
            # 使用所有token的平均作为整句表示
            sequence_output = bert_outputs["sequence_output"]  # [batch_size, seq_length, hidden_size]
            
            # 创建掩码，确保只对真实token进行平均，不包括padding
            mask = tf.cast(preprocessed["input_mask"], tf.float32)  # [batch_size, seq_length]
            mask = tf.expand_dims(mask, axis=-1)  # [batch_size, seq_length, 1]
            
            # 应用掩码并计算平均值
            masked_sum = tf.reduce_sum(sequence_output * mask, axis=1)  # [batch_size, hidden_size]
            token_count = tf.reduce_sum(mask, axis=1)  # [batch_size, 1]
            pooled_output = masked_sum / (token_count + 1e-10)  # 避免除以0
        else:  # "none"
            # 返回序列输出
            pooled_output = bert_outputs["sequence_output"]
        
        # 如果指定了输出维度，则应用投影
        if self.output_dim is not None and self.projection is not None:
            pooled_output = self.projection(pooled_output)
        
        return pooled_output

    def get_config(self):
        """获取层的配置，用于序列化"""
        base_config = super(BertEmbedding, self).get_config()
        base_config.update({"config": self.config})
        return base_config


if __name__ == "__main__":
    # 测试代码
    from tensorflow.keras.layers import Dense, InputLayer
    from tensorflow.keras.models import Sequential

    print("测试 BertEmbedding")

    # 构建模型
    def build_model():
        model = Sequential([
            InputLayer(input_shape=(), dtype=tf.string),
            BertEmbedding(config={
                "model_url": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2",  # 使用小型BERT便于测试
                "output_dim": 64,
                "max_seq_length": 64,
                "trainable": False
            }),
            Dense(1, activation="sigmoid"),
        ])
        return model

    # 测试数据
    input_data = tf.constant([
        "This is a test sentence for BERT embedding.",
        "Another example to check if the BERT embedding works."
    ])
    labels = tf.constant([1, 0])  # 假设是二分类标签
    
    try:
        # 构建并编译模型
        model = build_model()
        model.compile(
            optimizer="adam", 
            loss="binary_crossentropy", 
            metrics=["accuracy"]
        )
        
        # 打印模型摘要并训练
        model.summary()
        model.fit(input_data, labels, epochs=1)
        
        # 测试预测
        predictions = model.predict(input_data)
        print("预测结果:", predictions)
        print("测试成功")
    except Exception as e:
        print(f"测试失败: {str(e)}") 