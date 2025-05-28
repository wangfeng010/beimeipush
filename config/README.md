# 配置文件说明

- 特征预处理配置（`feat.yml`）
- 部署文件配置（`deploy.yml`）：用于线上部署时推荐引擎取数。
- 数据集配置（`data.yml`）：离线训练时的数据集配置，包含文件地址，数据列名称，数据列类型等。
- 训练配置（`train.yml`）：模型训练时的参数，包含学习率、batch size、Epoch等

## 特征预处理配置

特征预处理配置用于定义如何对原始数据进行处理以生成输入模型的特征。其中YAML用于存储配置，见`config/feat.yml`。另外，`light_ctr/utils/feat_config.py`中实现对应逻辑结构，用于校验。

具体而言，特征预处理配置主要包含以下几个部分：

1. **OperationConfig**:
   - **描述**: 定义单个算子的配置参数。
   - 属性
     - `col_in`: 输入列名，可以是单个字符串或列表。
     - `col_out`: 输出列名。
     - `func_name`: 算子对应的函数名，该函数位于 `operation.py` 中。
     - `func_parameters`: 函数的额外参数，可选。

2. **PipelineConfig**:
   - **描述**: 定义一个数据处理管道的配置参数。
   - 属性
     - `feat_name`: 输入模型的特征名。
     - `feat_type`: 特征的数据类型，如 `varlen_sparse`, `sparse`, `dense`。
     - `operations`: 一系列 `OperationConfig` 对象，表示该管道中的一系列操作。
     - `input_sample`: 输入样本的示例数据，可选。
     - `func_list`: 可调用函数列表，可选。

   - 方法
     - `__post_init__`: 初始化后处理方法，将 `operations` 中的每个 `OperationConfig` 对象实例化，并确保 `func_name` 存在于 `SINGLE_PROCESSOR_DICT` 或 `CROSS_PROCESSOR_DICT` 中。如果 `func_name` 是 `AsType`，则将其 `func_parameters` 中的 `target_dtype` 替换为实际的 TensorFlow 数据类型。

3. **ProcessConfig**:

   - **描述**: 定义整个特征处理过程的配置。
   - 属性
     - `pipelines`: 一系列 `PipelineConfig` 对象，表示整个处理过程中的一系列数据管道。

   - 方法
     - `__post_init__`: 初始化后处理方法，将 `pipelines` 中的每个 `PipelineConfig` 对象实例化。

### 示例

假设我们有一个特征处理配置文件 `temp.yml`，内容如下：

```yaml
pipelines:
  - feat_name: prod_buy_cnt_emb
    feat_type: SparseFeature
    operations:
      - col_in: prod_buy_cnt
        col_out: prod_buy_cnt
        func_name: FillNaString
        func_parameters:
          fill_value: "0#0"
      - col_in: prod_buy_cnt
        col_out: prod_buy_cnt_emb # col_in = col_out 表示原地操作
        func_name: SplitEmbedding
        func_parameters:
          first_sep: "|"
          second_sep: "#"
          second_sep_pos: 0
          padding_value: "0"
          second_sep_item_num: 2
          max_length: 5
          embedding_dim: 8
          vocab_size: 1000
          pooling: "mean"
```

在代码中，可以这样加载和解析这个配置文件：

```python
import yaml
from light_ctr.utils.feat_config import ProcessConfig

with open("/path/to/feat.yml", "r") as f:
    config = yaml.safe_load(f)

cfg = ProcessConfig(**config)
```

### 逻辑解释

该配置文件定义了一个数据处理管道（pipeline），用于对名为 `prod_buy_cnt_emb` 的特征进行预处理。具体来说，这个管道包含两个处理步骤（operations），它们分别应用不同的函数来处理特征。以下是每个部分的详细解释：

1. **特征名称和类型**：

   ```yaml
   feat_name: prod_buy_cnt_emb
   feat_type: SparseFeature
   ```

   - `feat_name`: 特征的名称，这里命名为 `prod_buy_cnt_emb`。
   - `feat_type`: 特征的数据类型，这里指定为 `SparseFeature`，表示这是一个稀疏特征。

2. **处理步骤（Operations）**： 每个处理步骤由 `col_in`、`col_out`、`func_name` 和 `func_parameters` 组成。

   - **第一步：FillNaString**

     ```yaml
     - col_in: prod_buy_cnt
       col_out: prod_buy_cnt
       func_name: FillNaString
       func_parameters:
         fill_value: "0#0"
     ```

     - `col_in`: 输入列的名称，这里是 `prod_buy_cnt`。

     - `col_out`: 输出列的名称，这里是 `prod_buy_cnt`。这表示处理结果会覆盖原来的列。

     - `func_name`: 要调用的函数名称，这里是 `FillNaString`，用于填充缺失值。

     - ```
       func_parameters
       ```

       : 函数的参数，具体如下：

       - `fill_value`: 用于填充缺失值的字符串，这里是 `"0#0"`。

   - **第二步：SplitEmbedding**

     ```yaml
     - col_in: prod_buy_cnt
       col_out: prod_buy_cnt_emb
       func_name: SplitEmbedding
       func_parameters:
         first_sep: "|"
         second_sep: "#"
         second_sep_pos: 0
         padding_value: "0"
         second_sep_item_num: 2
         max_length: 5
         embedding_dim: 8
         vocab_size: 1000
         pooling: "mean"
     ```

     - `col_in`: 输入列的名称，这里是 `prod_buy_cnt`。

     - `col_out`: 输出列的名称，这里是 `prod_buy_cnt_emb`。这表示处理结果将存储在一个新的列中。

     - `func_name`: 要调用的函数名称，这里是 `SplitEmbedding`，用于将字符串拆分成嵌入向量。

     - ```
       func_parameters
       ```

       : 函数的参数，具体如下：

       - `first_sep`: 第一级分隔符，这里是 `"|"`。
       - `second_sep`: 第二级分隔符，这里是 `"#"`。
       - `second_sep_pos`: 第二级分隔符的位置，这里是 `0`，表示使用第零个位置的分隔符。
       - `padding_value`: 填充值，这里是 `"0"`。
       - `second_sep_item_num`: 第二级分隔符分隔的项数，这里是 `2`。
       - `max_length`: 最大长度，这里是 `5`。
       - `embedding_dim`: 嵌入维度，这里是 `8`。
       - `vocab_size`: 词汇表大小，这里是 `1000`。
       - `pooling`: 池化方法，这里是 `"mean"`，表示使用均值池化。

总结起来，这个配置文件定义了一个数据处理管道，其中包含两个步骤：首先填充缺失值，然后将字符串拆分成嵌入向量。处理后的结果存储在 `prod_buy_cnt_emb` 列中。


## 线上部署配置

线上部署时需要向推荐引擎提供配置文件，说明需要向排序模型提供哪些数据。配置文件的例子见`deploy.yml`。配置文件描述了从不同数据源获取特征字段的配置。每个数据源包括用户（user）、项目（item）和上下文（context）。每个数据源中的特征字段通过 qualifier 和 field 进行映射，qualifier 是从数据源取数时的键，field 是在模型内部使用的字段名称。

配置格式：
```
raw_data:
  sourcesides:
    - sourceside: <数据源类型>
      offlineFeature:
        column_qualifiers:
          - qualifier: <数据源字段键>
            field: <模型内字段名>
        # 可选配置
        source: <数据存储类型>
        column_family: <列族名>
        request_context:
          - qualifier: <请求上下文字段键>
            field: <模型内字段名>
        item_context:
          <页面类型>:
            - qualifier: <项目上下文字段键>
              field: <模型内字段名>
```
### 解释

1. **`raw_data`**: 根节点，包含所有数据源的配置。
2. **`sourcesides`**: 列表，每个元素代表一个数据源。
3. **`sourceside`**: 数据源类型，例如 `user`、`item` 或 `context`。
4. `offlineFeature`
   : 存储离线特征的配置。
   - **`source`**: 数据存储类型，可选，例如 `hbase`。
   - **`column_family`**: 列族名，仅在 `source` 为 `hbase` 时使用。
   - `column_qualifiers`: 列表，每个元素是一个特征字段的映射。
     - **`qualifier`**: 从数据源取数时的键。
     - **`field`**: 在模型内部使用的字段名。
5. **`request_context`**: 请求上下文的配置，仅在 `sourceside` 为 `context` 时使用。
6. `item_context`: 项目上下文的配置，仅在`sourceside`为`context` 时使用。
   - **`<页面类型>`**: 项目上下文的具体页面类型，例如 `item_streaming_page`、`item_streaming_feed` 或 `item_streaming_lungutang`，一般没有实时特征则不填。
   - **`column_qualifiers`**: 与 `offlineFeature` 中的 `column_qualifiers` 类似，用于映射项目上下文的特征字段。