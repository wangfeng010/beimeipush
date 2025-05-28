import tensorflow as tf


class Expert(tf.keras.layers.Layer):
    """
    专家层，用于在MMoE模型中处理特定任务的特征。

    @param units: 该层输出的单元数。
    @param activation: 激活函数，默认为ReLU。
    """

    def __init__(self, units: int, activation: str = "relu", **kwargs):
        super(Expert, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(self.units, activation=self.activation)
        super(Expert, self).build(input_shape)

    def call(self, inputs):
        """
        前向传播方法，将输入传递给密集层并返回结果。

        @param inputs: 输入数据。
        @return: 经过密集层处理后的输出。
        """
        return self.dense(inputs)

    def get_config(self):
        config = super(Expert, self).get_config()
        config.update({"units": self.units, "activation": self.activation})
        return config


class Gate(tf.keras.layers.Layer):
    """
    门控层，用于在MMoE模型中决定每个专家的权重。

    @param num_experts: 专家的数量。
    @param gate_units: 门控层中间层的单元数。
    """

    def __init__(
        self, num_experts: int, gate_units: int, dropout: float = 0.1, **kwargs
    ):
        super(Gate, self).__init__()
        self.num_experts = num_experts
        self.gate_units = gate_units
        self.dropout = dropout

    def build(self, input_shape):
        self.hidden = tf.keras.layers.Dense(self.gate_units, activation="relu")
        self.output_layer = tf.keras.layers.Dense(
            self.num_experts, activation="softmax"
        )
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout)
        super(Gate, self).build(input_shape)

    def call(self, inputs):
        """
        前向传播方法，将输入传递给密集层并返回结果。

        @param inputs: 输入数据。
        @return: 经过密集层处理后的输出，表示每个专家的权重。
        """
        hidden_output = self.hidden(inputs)
        return self.dropout(self.output_layer(hidden_output))

    def get_config(self):
        config = super(Gate, self).get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "gate_units": self.gate_units,
                "dropout": self.dropout,
            }
        )
        return config


class MMoE(tf.keras.layers.Layer):
    """
    多任务多专家（MMoE）模型，用于同时处理多个任务。

    @param num_experts: 专家的数量。
    @param num_tasks: 任务的数量。
    @param expert_units: 每个专家层的输出单元数。
    @param gate_units: 每个门控层的输出单元数。
    @param dropout: 门控层的dropout率。
    @param activation: 专家层的激活函数，默认为ReLU。

    @inputs
        - inputs (tensor): 输入数据。shape = (batch_size, num_features)

    @outputs
        - outputs (tensor): 模型的输出，表示每个任务的预测结果。
        shape = (batch_size, num_tasks)
    """

    def __init__(
        self,
        num_experts: int,
        num_tasks: int,
        expert_units: int,
        gate_units: int,
        dropout: float = 0.1,
        activation: str = "relu",
        **kwargs,
    ):
        super(MMoE, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.expert_units = expert_units
        self.gate_units = gate_units
        self.dropout = dropout
        self.activation = activation

    def build(self, input_shape):
        self.experts = [
            Expert(self.expert_units, self.activation) for _ in range(self.num_experts)
        ]
        self.gates = [
            Gate(self.num_experts, self.gate_units, dropout=self.dropout)
            for _ in range(self.num_tasks)
        ]
        super(MMoE, self).build(input_shape)

    def call(self, inputs):
        """
        前向传播方法，计算MMoE模型的输出。

        @param inputs: 输入数据。
        @return: 每个任务的输出。
        """
        expert_outputs = [expert(inputs) for expert in self.experts]
        gate_outputs = [gate(inputs) for gate in self.gates]
        mmoe_outputs = []

        for gate_output in gate_outputs:
            # 增加一个维度以进行广播
            gate_output_expanded = tf.expand_dims(gate_output, axis=1)
            weighted_experts = [
                gate_output_expanded[:, :, i] * expert_output
                for i, expert_output in enumerate(expert_outputs)
            ]
            mmoe_outputs.append(
                tf.reduce_sum(tf.stack(weighted_experts, axis=-1), axis=-1)
            )

        return mmoe_outputs

    def get_config(self):
        config = super(MMoE, self).get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "num_tasks": self.num_tasks,
                "expert_units": self.expert_units,
                "gate_units": self.gate_units,
                "dropout": self.dropout,
                "activation": self.activation,
            }
        )
        return config


if __name__ == "__main__":
    # 创建一个完整的模型
    import numpy as np

    def create_mmoel_model(input_dim, num_experts, num_tasks, expert_units, gate_units):
        inputs = tf.keras.Input(shape=(input_dim,))
        mmoe_outputs = MMoE(num_experts, num_tasks, expert_units, gate_units)(inputs)

        # 为每个任务添加一个输出层
        task_outputs = []
        for i in range(num_tasks):
            task_output = tf.keras.layers.Dense(
                1, activation="sigmoid", name=f"task_{i+1}_output"
            )(mmoe_outputs[i])
            task_outputs.append(task_output)

        model = tf.keras.Model(inputs=inputs, outputs=task_outputs)
        return model

    # 测试模型
    input_dim = 64
    num_experts = 4
    num_tasks = 2
    expert_units = 128
    gate_units = 32

    model = create_mmoel_model(
        input_dim, num_experts, num_tasks, expert_units, gate_units
    )

    # 编译模型
    model.compile(
        optimizer="adam", loss=["binary_crossentropy"] * num_tasks, metrics=["accuracy"]
    )

    # 生成随机数据
    X_train = np.random.uniform(size=(1000, input_dim))
    y_train = np.random.randint(2, size=(1000, num_tasks))

    # 将 y_train 分解成列表
    y_train_list = [y_train[:, i] for i in range(num_tasks)]

    # 训练模型
    model.fit(X_train, y_train_list, epochs=5, batch_size=32, validation_split=0.2)

    # 生成测试数据
    X_test = np.random.uniform(size=(200, input_dim))
    y_test = np.random.randint(2, size=(200, num_tasks))

    # 将 y_test 分解成列表
    y_test_list = [y_test[:, i] for i in range(num_tasks)]

    # 评估模型
    results = model.evaluate(X_test, y_test_list)
    print("Test Loss and Accuracy:", results)
