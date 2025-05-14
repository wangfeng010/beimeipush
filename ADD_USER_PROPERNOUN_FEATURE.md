# 添加用户实体词偏好(user_propernoun)特征 - TodoList

## 概述
本文档描述了向推送排序系统中添加用户实体词偏好(user_propernoun)特征的完整流程。该特征存储在数据库的`db_dws.dws_crd_lb_v_dd`表中，对应代码为'BM230'，格式类似"apple#1.02|nike#1.02"。

## 修改步骤

### 1. 修改SQL查询文件
- [x] 更新`data/user.sql`中的SQL查询，添加对BM230代码的查询
- [x] 更新`data/train.sql`中的SQL查询，添加对BM230代码的查询
- [x] 更新`data/new_user.sql`中的SQL查询，添加对BM230代码的查询

### 2. 更新配置文件
- [x] 在`config/config.yml`中添加新特征的处理pipeline
- [x] 在`config/infer.yml`中的data_columns列表中添加新字段

### 3. 验证与测试
- [ ] 确保特征处理函数能够正确处理新特征格式
- [ ] 重新训练模型
- [ ] 验证新特征是否被正确融入模型

## 详细修改说明

### 1. SQL查询修改
在`data/user.sql`、`data/train.sql`和`data/new_user.sql`文件中，在查询`db_dws.dws_crd_lb_v_dd`表的部分添加对'BM230'代码的查询：

1. 修改查询条件：
```sql
where
    p_date = '{now}'
    and l_code in ('BM79', 'BM176', 'BM210', 'BM55', 'BM70', 'BM230')
    and person_id > 0
```

2. 在SELECT部分添加新字段：
```sql
max(
    case
        when l_code = 'BM230'
        and l_value is not null then l_value
        else null
    end
) as user_propernoun
```

### 2. 配置文件修改

#### config.yml修改
添加新特征的处理pipeline：
```yaml
- feat_name: user_propernoun_hash
  feat_type: varlen_sparse
  vocabulary_size: 10000
  embedding_dim: 8
  input_sample: "apple#1.02|nike#1.02"
  operations:
    - col_in: user_propernoun
      col_out: user_propernoun
      func_name: fillna
      func_parameters:
        na_value: "null#0"
    - col_in: user_propernoun
      col_out: user_propernoun
      func_name: split
      func_parameters:
        sep: "|"
    - col_in: user_propernoun
      col_out: user_propernoun
      func_name: seperation
      func_parameters:
        sep: "#"
    - col_in: user_propernoun
      col_out: user_propernoun_code
      func_name: list_get
      func_parameters:
        item_index: 0
    - col_in: user_propernoun_code
      col_out: user_propernoun_code
      func_name: padding
      func_parameters:
        max_len: 5
        pad_value: "null"
    - col_in: user_propernoun_code
      col_out: user_propernoun_code_hash
      func_name: list_hash
      func_parameters:
        vocabulary_size: 10000
```

#### infer.yml修改
更新data_columns列表：
```yaml
data_columns:
  - user_id
  - watchlists
  - holdings
  - country
  - prefer_bid
  - user_propernoun
```

### 3. 验证与测试
1. 执行训练脚本：`python train.py`
2. 启动推理服务：`uvicorn infer:app --reload --port=12333`
3. 使用测试数据验证特征是否被正确处理

## 可能的问题和解决方案
1. 如果新特征格式与现有处理函数不兼容，可能需要在`utils/preprocess.py`中添加新的处理函数
2. 如果特征值缺失，确保fillna操作能够提供合适的默认值
3. 如果特征处理pipeline出错，检查config.yml中的配置是否正确 