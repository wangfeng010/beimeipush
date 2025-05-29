from typing import Dict, List, Optional, Any

import tensorflow as tf
from tensorflow.keras.layers import Layer


class StringConcat(Layer):
    """
    字符串拼接层，将多个字符串特征拼接成单个字符串。
    
    此层接收一个字符串列表作为输入，并将它们按照指定的分隔符拼接成单个字符串。
    
    Attributes:
        sep (str): 用于拼接字符串的分隔符，默认为空格。
    """

    def __init__(
        self,
        sep: str = " ",
        **kwargs,
    ):
        """
        初始化字符串拼接层。

        Args:
            sep (str): 用于拼接字符串的分隔符，默认为空格。
            **kwargs: 传递给父类的其他参数。
        """
        super(StringConcat, self).__init__(**kwargs)
        self.sep = sep

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        """
        执行字符串拼接操作。

        Args:
            inputs: 输入的字符串张量列表。

        Returns:
            拼接后的字符串张量。
        """
        if not isinstance(inputs, list):
            return inputs
        
        # 初始化拼接结果为第一个张量
        result = inputs[0]
        
        # 逐个拼接其余张量
        for i in range(1, len(inputs)):
            result = tf.strings.join([result, inputs[i]], separator=self.sep)
        
        return result

    def get_config(self):
        """获取层的配置，用于序列化。"""
        config = super(StringConcat, self).get_config()
        config.update({"sep": self.sep})
        return config


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model

    print("测试 StringConcat")

    # 构建测试模型
    input1 = Input(shape=(), dtype=tf.string, name="input1")
    input2 = Input(shape=(), dtype=tf.string, name="input2")
    output = StringConcat(sep=" ")([input1, input2])
    model = Model(inputs=[input1, input2], outputs=output)

    # 测试数据
    test_input1 = np.array(["Hello", "Test"])
    test_input2 = np.array(["World", "Case"])
    
    # 运行模型
    result = model.predict([test_input1, test_input2])
    print("输入1:", test_input1)
    print("输入2:", test_input2)
    print("拼接结果:", result)
    print("测试成功") 