# 配置模板说明

配置模板可参考`templates/demo_config.yml`。

## 1. 完整配置

当前版本的完整配置为：

- `datasets`：用于配置数据集的路径、数据列名等。
- `process`：用于配置普通单列数据的预处理。
- `interactions`：用于配置特征交叉。
- `label_process`：用于配置真实标签的预处理。
- `model`：用于配置模型相关的参数，如嵌入层的词典大小与嵌入维度。
上述配置由`uniprocess/config.py`中的`Config`类来加载。

## 2. 数据集配置

数据集根据模型的阶段拆分：

- `trainset`：训练集
- `valset`：验证集

上述配置由`uniprocess/config.py`中的`DataSetsConfig`类来加载，其中每个子集均由`DatasetConfig`加载。

每个子集的配置项为：

- `data_dir`：原始数据目录，支持文件夹与文件。默认转化为`data_path`，筛选`.txt`文件和`.csv`文件。读取配置时请读取`data_path`。
- `sep`：csv文件每列数据之间的分隔符。
- `raw_columns`：每个数据列的列名组成的列表。
- `chunksize`：pandas版本实现时，每次读取的行数。当前版本采用Python自带的`csv`包读取，是逐行读取，暂时未使用。
- `header`：数据文件的第一行是否是列名。
- `file_num`: 最近N天的数据文件。若为None，则默认是全部文件。
- `label_columns`：真实标签的名称列表。应该为`raw_columns`的子集。

## 3. 普通预处理

普通预处理是指仅有一个输入的预处理。该配置通过`uniprocess/config.py`中的`ProcessConfig`类来加载。

配置项为：

- `pipelines`: 包含多个数据管道的列表，一个数据管道一般实现了针对一列数据进行的连续的预处理。例如：一个数据管道要处理年龄这个特征，需要进行缺失值填充、年龄分段两个预处理。
- `embedding_dim`：稀疏特征的嵌入维度。
- `pooling_type`：变长稀疏特征的池化类型。

### 3.1 数据管道

数据管道中包含了一系列预处理算子，可以将原始数据转为为一个模型可用的特征。该配置通过`uniprocess/config.py`中的`PipelineConfig`类来加载。

配置项为：

- `feat_name`： 输入模型的特征名
- `feat_type`：特征的类型，可以为`varlen_sparse`, `sparse`, `dense`
- `operations`：预处理算子，一般是针对基础数据类型的最小操作。通过不同预处理算子的组合可以实现复杂的预处理。
- `vocabulary_size`：词表大小，仅针对`varlen_sparse`, `sparse`类型的数据。
- `embedding_dim`：稀疏特征的嵌入维度，仅针对`varlen_sparse`, `sparse`类型的数据。
- `input_sample`：输入的示例数据，帮助debug。

#### 3.1.1 算子

算子的配置由`uniprocess/config.py`中的`OperationConfig`类来加载。

配置项为：

- `col_in`：输入数据列的名称，可以为单个字符串，也支持列表。
- `col_out`：输出数据列的名称。如果与`col_in`相同，表示原地操作。但是每次都设置不同的输出名，会导致内存占用过高。
- `func_name`：算子的函数名。具体见[算子介绍](uniprocess/operations/README.md)。
- `func_parameters`：算子函数的参数。具体见[算子介绍](uniprocess/operations/README.md)。
