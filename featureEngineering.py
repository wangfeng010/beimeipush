# 架构概述
# 原始数据存放
#     线上：data/train/*txt
#     线下：data/train/*csv
# 特征工程处理流程的定义
#     树模型：config.yml
#     深度模型：feat.yml

# 01 环境导入 —— 确定项目的 核心架构 和 工具
import os
import sys
import json
from functools import partial
from datetime import datetime
from hashlib import md5
from typing import Any, Dict, List, Union
import pandas as pd
import yaml
import numpy as np
from pprint import pprint
from glob import glob  # 添加glob模块导入
import time


# 添加项目路径
project_root = os.getcwd()
env_path = os.path.join(project_root, 'env')
if env_path not in sys.path:
    sys.path.insert(0, env_path)

# 02 加载原始数据
data_path = 'data/train/*.csv'
# 使用glob获取所有匹配的CSV文件路径
csv_files = glob(data_path)
if not csv_files:
    raise ValueError(f"No CSV files found in {data_path}")

# 读取并合并所有CSV文件
df_raw = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
for col in df_raw.columns: print(f"{col}: {df_raw[col].iloc[0]}") # 直观展示各列的具体数据的样式
# 我们可以看到原始数据的样子 用pd读去后
# user_id	create_time	log_type	watchlists	holdings	country	prefer_bid	user_propernoun	push_title	push_content	item_code	item_tags	submit_type
# 0	1800000316	2025-05-30 09:30:14	PR	VIXY_171 & UPST_185 & AMD_185 & ASTS_185 & SHO...	RKLB,186|NFLX,185|ISRG,185|NVDA,185|OKLO,169|A...	China	TWCUX#0.54|ACFOX#0.53721|ALAB#0.38253|SPOT#0.3...	NaN	Ainvest Newswire	Nvidia Corporation shares rise 3.25% intraday ...	[{"market":"185","score":1,"code":"NVDA","tagI...	[{"score":0.9049434065818787,"tagId":"51510","...	autoFlash
# 1	1800000316	2025-05-30 09:30:14.995	PR	VIXY_171 & UPST_185 & AMD_185 & ASTS_185 & SHO...	RKLB,186|NFLX,185|ISRG,185|NVDA,185|OKLO,169|A...	China	TWCUX#0.54|ACFOX#0.53721|ALAB#0.38253|SPOT#0.3...	NaN	Ainvest Newswire	Nvidia Corporation shares rise 3.25% intraday ...	[{"market":"185","score":1,"code":"NVDA","tagI...	[{"score":0.9049434065818787,"tagId":"51510","...	autoFlash
# 2	1800001318	2025-05-30 19:49:03	PR	IBIT_185 & TEM_185 & DLR_169 & HOOD_185 & ETH_...	BRK.B,169|HBAN,185	China	BRK.B#0.54|SQQQ#0.28932|IBIT#0.24075|HBAN#0.17851	NaN	Breaking News	Fed's Daly Says Still Comfortable With Two Rat...	NaN	[{"score":0,"tagId":"56127","name":"us_stock",...	flash
# 3	1800001318	2025-05-30 05:09:09	PR	IBIT_185 & TEM_185 & DLR_169 & HOOD_185 & ETH_...	BRK.B,169|HBAN,185	China	BRK.B#0.54|SQQQ#0.28932|IBIT#0.24075|HBAN#0.17851	NaN	UP Fintech's Q1 Surge: A C...	The financial markets are always searching for...	[{"market":"185","score":0,"code":"TIGR","tagI...	[{"score":0.4888981282711029,"code":"us_low_im...	NaN
# 4	1800001324	2025-05-30 13:57:50	PR	NaN	NaN	United States	NaN	blackwell nvl72 ai#3.06|nvidia#3.06	Trump Media's $2.32 Billio...	Trump Media's recent acquisition of $2.32 bill...	[{"market":"185","score":0,"code":"DJT","tagId...	[{"score":0.6,"code":"Stock","tagId":"1000147"...	NaN

# 我们可以做基础的统计分析
# 原始数据pd的形状（多少行 多少列）：df_raw.shape
# 每一列的列名：list(df_raw.columns)
# 直观展示一个数据：df_raw.head()
# 直观展示各列的具体数据的样式：for col in df_raw.columns: print(f"{col}: {df_raw[col].iloc[0]}")
#                          也就是col 和 df_raw[col].iloc[0]
# user_id: 1800000316
# create_time: 2025-05-30 09:30:14
# log_type: PR
# watchlists: VIXY_171 & UPST_185 & AMD_185 & ASTS_185 & SHOP_185 & RKLB_186 & IBKR_185 & PLTR_185 & ISRG_185 & APP_185 & VIXM_171 & NFLX_185 & NVDA_185 & ORCL_169 & QCOM_185 & SMCI_185 & SONY_169 & SOUN_185 & TSLA_185 & TSM_169 & U_169 & UBER_169 & AAPL_185 & ALAB_185 & AMT_169 & AMZN_185 & ARM_185 & BABA_169 & BRK.A_169 & COST_185 & DELL_169 & DJT_185 & DOCU_185 & EQIX_185 & EWBC_185 & GME_169 & GOOGL_185 & HOOD_185 & INTC_185 & LULU_185 & META_185 & MSFT_185 & OKLO_169 & IONQ_169 & NNE_186 & SPOT_169
# holdings: RKLB,186|NFLX,185|ISRG,185|NVDA,185|OKLO,169|ALAB,185|CSCO,185|MSFT,185|SPOT,169|TSLA,185|AAPL,185|AMZN,185|GOOGL,185|HOOD,185|PLTR,185|ASML,185|AVGO,185|COST,185|META,185
# country: China
# prefer_bid: TWCUX#0.54|ACFOX#0.53721|ALAB#0.38253|SPOT#0.34926|NVDA#0.34051
# user_propernoun: nan
# push_title: Ainvest Newswire
# push_content: Nvidia Corporation shares rise 3.25% intraday after Q1 2026 revenue beats expectations at $44.1B.
# item_code: [{"market":"185","score":1,"code":"NVDA","tagId":"U000017465","name":"英伟达","type":0,"parentId":"US_ROBOT7140bd7a307047b2"}]
# item_tags: [{"score":0.9049434065818787,"tagId":"51510","name":"us_high_importance","type":4,"parentId":"US_ROBOT7140bd7a307047b2"},{"score":0.9049434065818787,"tagId":"53111","name":"aigc_today_mover","type":4,"parentId":"US_ROBOT7140bd7a307047b2"},{"tagId":"1002","name":"no_penny_stock","type":4,"parentId":"US_ROBOT7140bd7a307047b2"}]
# submit_type: autoFlash

# 03 加载并解析YAML配置 —— "蓝图"文件
print("📋 配置文件使用说明:")
print("  🌳 树模型: config.yml (包含特征工程 + 训练参数)")
print("  🧠 深度模型: feat.yml (特征工程) + train.yml (训练参数)")

# 加载树模型配置 (config.yml 包含特征和训练配置)
with open('config/config.yml', 'r', encoding='utf-8') as f:
    config_yml = yaml.safe_load(f)

# 从 config.yml 提取树模型的特征配置
tree_feat_config = config_yml['features']

# 加载深度模型的特征配置 (feat.yml)
with open('config/feat.yml', 'r', encoding='utf-8') as f:
    deep_feat_config = yaml.safe_load(f)

# 当前使用树模型的特征配置进行演示
feat_config = tree_feat_config

print(feat_config)
# 核心就是理解这个PyYAML库提供的yaml.safe_load()工具的使用
# 读取一个 YAML 格式的文件或字符串，并将其转换为 Python 对象（通常是字典或列表）
# 这里的feat_config就变成了这样
# 这是一个两层结构的字典：
# 顶层有两个键：exclude_features 和 pipelines
# pipelines 是一个列表，每个元素都是一个包含特征处理配置的字典
# 所以针对这个处理出来的数据结构 len(feat_config['pipelines']) 是可以看一下有多少个在蓝图中写着的需要处理的特征
# {
#     # 第一部分：特征排除配置
#     'exclude_features': {
#         'current': 'default',
#         'default': [],
#         'exclude_user_behavior': [
#             'user_watch_stk_code',
#             'prefer_bid_code',
#             'hold_bid_code',
#             'user_propernoun'
#         ],
#         'exclude_user_propernoun': [
#             'user_propernoun'
#         ]
#     },

#     # 第二部分：特征处理流水线列表
#     'pipelines': [
#         # 每个流水线的基本结构
#         {
#             'embedding_dim': <int>,          # 嵌入维度
#             'feat_name': <str>,              # 特征名称
#             'feat_type': <str>,              # 特征类型
#             'input_sample': <str>,           # 输入样例
#             'vocabulary_size': <int>,        # 词汇表大小
#             'operations': [                  # 操作列表
#                 {
#                     'col_in': <str>,         # 输入列
#                     'col_out': <str>,        # 输出列
#                     'func_name': <str>,      # 函数名
#                     'func_parameters': {     # 函数参数
#                         <param_name>: <value>
#                     }
#                 },
#                 # ... 可以有多个操作
#             ]
#         },
#         # ... 可以有多个流水线
#     ]
# }

# 这里需要补充介绍的就是yaml.safe_load(f)函数是怎么解析的流程
# 1. 词法分析
#     识别
#         缩进级别
#         标点符号（：，-，等）
#         标量值（str，数字，bool）
#         换行符
# 2. 语法分析
#     识别
#         Mapping（使用：结构）
#             number: 123        # 转换为 int
#             float: 123.45     # 转换为 float
#             boolean: true     # 转换为 bool
#             string: "hello"   # 转换为 str
#             null_value: null  # 转换为 None
#         Sequence（使用 - 开头的列表项）
#         嵌套结构（通过缩进表示层级关系）

# 我们核心是想看一下一个pipeline中的一个特征 他被解析成了什么结构
# 适配 config.yml 和 feat.yml 的不同结构
if 'pipelines' in feat_config:
    # feat.yml 格式
    example_pipeline = feat_config['pipelines'][0]
elif 'process' in feat_config and 'pipelines' in feat_config['process']:
    # config.yml 格式
    example_pipeline = feat_config['process']['pipelines'][0]
else:
    print("⚠️ 无法找到特征配置格式")
    example_pipeline = None

if example_pipeline:
    pprint(example_pipeline)
# {'embedding_dim': 8,
#  'feat_name': 'hour',
#  'feat_type': 'sparse',
#  'input_sample': '2024-08-02 00:44:05',
#  'operations': [{'col_in': 'create_time',
#                  'col_out': 'create_time',
#                  'func_name': 'fillna',
#                  'func_parameters': {'na_value': '2024-08-02 00:16:34'}},
#                 {'col_in': 'create_time',
#                  'col_out': 'hour',
#                  'func_name': 'to_hour',
#                  'func_parameters': {}}],
#  'vocabulary_size': 24}

# 04 定义特征操作函数
# 定义一个单独的py文件来实现这个操作 并构建操作中心
# 这步骤的流程就是解耦成原子操作版本的Hugging Face框架里的
# - 🌟批量处理所有分割
#     def preprocess(examples):
#         return tokenizer(examples['text'], truncation=True) # 是否截断

#     processed_datasets = datasets.map(preprocess, batched=True)
# - 🌟当然也可以单样本处理
#     datasets = datasets.map(lambda x: {'length': len(x['text'])})
# 只是这里为了特征处理的通用性解耦成了一个一堆原子操作函数顺序执行的pipeline形式来拼成这个preprocess函数
# 并且这一堆函数的拼接顺序 和 每个函数传入什么参数 都用yaml文件 当作设计蓝图解耦的定义在外面
# 通过transform_func = partial(OP_HUB[func_name], **parameters) 的方式做成原子操作函数
# 然后
# if isinstance(col_in, list):
#     df[col_out] = df[col_in].apply(lambda row: transform_func(*row), axis=1)
# else:
#     df[col_out] = df[col_in].apply(transform_func)
# 的方式执行下去 相当于 datasets.map下去
# 区别是原来是一个函数 map下去完成处理 这里是一堆按蓝图传参和拼接原子函数 一个一个map下去

# 定义缺失值常量
MISSING_VALUE = [None, '', 'null', 'NULL', 'None', np.nan]

def fillna(x: Union[float, int, str], na_value: Union[float, int, str]) -> Union[float, int, str]:
    """填充缺失值"""
    if x in MISSING_VALUE or (isinstance(x, float) and pd.isna(x)):
        return na_value
    return x

def split(x: str, sep: str) -> List[str]:
    """字符串分割"""
    return str(x).split(sep)

def seperation(x: List[str], sep: str) -> List[List[str]]:
    """列表元素二次分割"""
    if not isinstance(x, list):
        return []
    return [item.split(sep) for item in x]

def list_get(x: List[List[Any]], item_index: int) -> List[Any]:
    """获取嵌套列表中指定位置的元素"""
    if not isinstance(x, list):
        return []
    result = []
    for sublist in x:
        if isinstance(sublist, list) and len(sublist) > item_index:
            result.append(sublist[item_index])
        else:
            result.append('null')
    return result

def remove_items(x: List[str], target_values: List[str]) -> List[str]:
    """移除列表中的指定元素"""
    if not isinstance(x, list):
        return []
    return [item for item in x if item not in target_values]

def padding(x: List[Any], pad_value: Union[str, float, int], max_len: int) -> List[Any]:
    """列表填充到指定长度"""
    if not isinstance(x, list):
        x = []
    if len(x) >= max_len:
        return x[:max_len]
    else:
        return x + [pad_value] * (max_len - len(x))

def list_hash(x: List[str], vocabulary_size: int) -> List[int]:
    """对列表中每个元素进行哈希"""
    if not isinstance(x, list):
        return []
    result = []
    for item in x:
        hash_val = int(md5(str(item).encode()).hexdigest(), 16) % vocabulary_size
        result.append(hash_val)
    return result

def str_hash(x: str, vocabulary_size: int) -> int:
    """字符串哈希"""
    return int(md5(str(x).encode()).hexdigest(), 16) % vocabulary_size

def to_hour(x: str) -> int:
    """提取时间中的小时"""
    try:
        dt = pd.to_datetime(x)
        return dt.hour
    except:
        return 0

def to_weekday(x: str) -> int:
    """提取时间中的星期"""
    try:
        dt = pd.to_datetime(x)
        return dt.weekday()
    except:
        return 0

def list_len(x: List) -> int:
    """列表长度"""
    if isinstance(x, list):
        return len(x)
    return 0

def int_max(x: int, max_value: int) -> int:
    """限制整数最大值"""
    return min(int(x), max_value)

def json_object_to_list(x: str, key: str) -> List[str]:
    """从JSON对象列表中提取指定键的值"""
    try:
        data = json.loads(x)
        if isinstance(data, list):
            return [item.get(key, 'null') for item in data if isinstance(item, dict)]
        return ['null']
    except:
        return ['null']

def map_to_int(x: Union[str, List], map_dict: Dict[str, int], default_code: int = 0) -> Union[List[int], int]:
    """映射到整数"""
    if isinstance(x, list):
        return [map_dict.get(item, default_code) for item in x]
    else:
        return map_dict.get(str(x), default_code)

# 构建操作中心 (OP_HUB)
OP_HUB = {
    'fillna': fillna,
    'split': split,
    'seperation': seperation,
    'list_get': list_get,
    'remove_items': remove_items,
    'padding': padding,
    'list_hash': list_hash,
    'str_hash': str_hash,
    'to_hour': to_hour,
    'to_weekday': to_weekday,
    'list_len': list_len,
    'int_max': int_max,
    'json_object_to_list': json_object_to_list,
    'map_to_int': map_to_int
}

print(f"OP_HUB 构建完成，包含 {len(OP_HUB)} 个操作函数")
print(f"可用函数: {list(OP_HUB.keys())}")

# 05 实现原子操作作用于df
def run_one_op(df: pd.DataFrame, operation: dict) -> pd.DataFrame:
    """执行单个特征操作"""
    # 获取操作配置
    col_in = operation['col_in']
    col_out = operation['col_out']
    func_name = operation['func_name']
    parameters = operation.get('func_parameters', {})
    
    # 检查函数是否存在
    if func_name not in OP_HUB:
        return df
    
    # 检查输入列是否存在
    input_cols = [col_in] if isinstance(col_in, str) else col_in
    if not all(col in df.columns for col in input_cols):
        return df
    
    # 准备特征转换函数
    transform_func = partial(OP_HUB[func_name], **parameters)
    
    # 执行特征转换
    if isinstance(col_in, list):
        df[col_out] = df[col_in].apply(lambda row: transform_func(*row), axis=1)
    else:
        df[col_out] = df[col_in].apply(transform_func)
    
    return df
#   1. 给他传入的是 operation
#   operations是list operation是dict
#  'operations': [{'col_in': 'create_time',
#                  'col_out': 'create_time',
#                  'func_name': 'fillna',
#                  'func_parameters': {'na_value': '2024-08-02 00:16:34'}},
#                 {'col_in': 'create_time',
#                  'col_out': 'hour',
#                  'func_name': 'to_hour',
#                  'func_parameters': {}}],
#  'operation' ： {'col_in': 'create_time',
#                  'col_out': 'create_time',
#                  'func_name': 'fillna',
#                  'func_parameters': {'na_value': '2024-08-02 00:16:34'}}

# 2. 先定义好数据检查
# input_cols = [col_in] if isinstance(col_in, str) else col_in
# 这句话的作用是
# 情况1：单列输入
# col_in = "user_id"
# input_cols = ["user_id"]  # 转成列表
# # 情况2：多列输入
# col_in = ["first_name", "last_name"]
# input_cols = ["first_name", "last_name"]  # 保持不变

# 3. 定义处理函数
# transform_func = partial(OP_HUB[func_name], **parameters)
# 从 OP_HUB 字典中获取名为 func_name 的函数
# 使用 partial 把函数参数 parameters 固定进去
# # 假设有这样一个函数在 OP_HUB 中
# def fillna(value, na_value="null"):
#     return value if value else na_value
# # 配置是这样的
# func_name = "fillna"
# parameters = {"na_value": "unknown"}
# # partial 后相当于
# transform_func = lambda value: fillna(value, na_value="unknown")

# 4. 定义处理函数怎么作用于数据集
# if isinstance(col_in, list): # 支持多列操作
#     df[col_out] = df[col_in].apply(lambda row: transform_func(*row), axis=1)
# else: # 单列操作
#     df[col_out] = df[col_in].apply(transform_func)

# 单列输入：直接对这一列的每个值应用转换函数
# 多列输入：把多列的值作为参数传给转换函数
# *row 的作用是把行的值展开作为函数参数
# axis=1 表示按行处理而不是按列

# 假设数据是：
# 单列操作
# df = pd.DataFrame({
#     'name': ['Alice', None, 'Bob']
# })
# # 配置是：
# col_in = 'name'
# col_out = 'name_filled'
# func_name = 'fillna'
# parameters = {'na_value': 'unknown'}
# # 执行后相当于：
# df['name_filled'] = df['name'].apply(lambda x: x if x else 'unknown')
# # 结果：['Alice', 'unknown', 'Bob']
# 多列操作
# # 假设数据是：
# df = pd.DataFrame({
#     'first_name': ['Alice', 'Bob', None],
#     'last_name': ['Smith', None, 'Johnson']
# })
# # 配置是：
# col_in = ['first_name', 'last_name']
# col_out = 'full_name'
# func_name = 'concat_names'
# parameters = {'sep': ' '}
# # 执行后相当于：
# df['full_name'] = df[['first_name', 'last_name']].apply(
#     lambda row: f"{row['first_name']} {row['last_name']}", 
#     axis=1
# )
# # 结果：['Alice Smith', 'Bob None', 'None Johnson']

# 06 实现原子操作拼成的完整process函数作用于df
def process_feature_pipelines(df_raw: pd.DataFrame, feat_config: dict) -> tuple[pd.DataFrame, list]:
    """执行特征工程流水线 - 适配不同的配置文件格式"""
    # 创建数据副本
    df = df_raw.copy()
    
    # 获取需要处理的流水线 - 适配不同格式
    if 'pipelines' in feat_config:
        # feat.yml 格式: { pipelines: [...] }
        pipelines = feat_config['pipelines']
    elif 'process' in feat_config and 'pipelines' in feat_config['process']:
        # config.yml 格式: { process: { pipelines: [...] } }
        pipelines = feat_config['process']['pipelines']
    else:
        print("⚠️ 无法找到特征配置中的 pipelines")
        return df, []

    # 记录成功处理的特征
    processed_features = []
    
    # 执行每个特征处理流水线
    for pipeline in pipelines:
        feat_name = pipeline['feat_name']
        operations = pipeline['operations']
        
        # 执行流水线中的每个操作
        for operation in operations:
            df = run_one_op(df, operation)

        # 记录处理成功的特征
        processed_features.append(feat_name)
    
    return df, processed_features

df_processed, processed_features = process_feature_pipelines(df_raw, feat_config)

# 07 分析处理结果
# 比较处理前后的数据结构
print("数据结构对比:")
print(f"原始列数: {len(df_raw.columns)}")
print(f"处理后列数: {len(df_processed.columns)}")
print(f"新增列数: {len(df_processed.columns) - len(df_raw.columns)}")

print("\n原始列名:")
print(list(df_raw.columns))

print("\n新增列名:")
new_columns = [col for col in df_processed.columns if col not in df_raw.columns]
print(new_columns)
# 数据结构对比（包含有很多中间特征列）
# 原始列数: 13
# 处理后列数: 30
# 新增列数: 17

# 原始列名:
# ['user_id', 'create_time', 'log_type', 'watchlists', 'holdings', 'country', 'prefer_bid', 'user_propernoun', 'push_title', 'push_content', 'item_code', 'item_tags', 'submit_type']

# 新增列名:
# ['hour', 'weekday', 'user_watch_stk_code', 'user_watch_stk_code_hash', 'country_hash', 'prefer_bid_code', 'prefer_bid_code_hash', 'hold_bid_code', 'hold_bid_code_hash', 'user_propernoun_code', 'user_propernoun_hash', 'push_title_hash', 'title_len', 'item_code_hash', 'submit_type_hash', 'tagIds', 'tag_id_hash']
# 查看成功生成的特征
print("成功生成的特征详情:")
for feat_name in processed_features:
    if feat_name in df_processed.columns:
        sample_data = df_processed[feat_name].iloc[0]
        data_type = type(sample_data).__name__
        print(f"  {feat_name}: {data_type} = {sample_data}")
# 成功生成的特征详情:
#   hour: int64 = 9
#   weekday: int64 = 4
#   user_watch_stk_code_hash: list = [5353, 3385, 2290, 7019, 6324]
#   country_hash: int64 = 106
#   prefer_bid_code_hash: list = [2449, 9133, 9212, 4180, 304]
#   hold_bid_code_hash: list = [1766, 9287, 1277, 304, 777]
#   user_propernoun_hash: list = [8381, 8381, 8381, 8381, 8381]
#   push_title_hash: int64 = 7
#   title_len: int64 = 14
#   item_code_hash: list = [304, 8381, 8381, 8381, 8381]
#   submit_type_hash: int64 = 6
#   tag_id_hash: list = [8139, 798, 880]
# 显示最终处理结果的部分数据
print("最终处理结果预览:")
display_cols = ['user_id', 'log_type'] + processed_features
display_cols = [col for col in display_cols if col in df_processed.columns]

print(df_processed[display_cols])
# user_id	log_type	hour	weekday	user_watch_stk_code_hash	country_hash	prefer_bid_code_hash	hold_bid_code_hash	user_propernoun_hash	push_title_hash	title_len	item_code_hash	submit_type_hash	tag_id_hash
# 0	1800000316	PR	9	4	[5353, 3385, 2290, 7019, 6324]	106	[2449, 9133, 9212, 4180, 304]	[1766, 9287, 1277, 304, 777]	[8381, 8381, 8381, 8381, 8381]	7	14	[304, 8381, 8381, 8381, 8381]	6	[8139, 798, 880]
# 1	1800000316	PR	9	4	[5353, 3385, 2290, 7019, 6324]	106	[2449, 9133, 9212, 4180, 304]	[1766, 9287, 1277, 304, 777]	[8381, 8381, 8381, 8381, 8381]	7	14	[304, 8381, 8381, 8381, 8381]	6	[8139, 798, 880]
# 2	1800001318	PR	19	4	[898, 5602, 5053, 8877, 5282]	106	[9153, 3036, 898, 6687, 8381]	[9153, 6687, 8381, 8381, 8381]	[8381, 8381, 8381, 8381, 8381]	7	13	[8381, 8381, 8381, 8381, 8381]	4	[4634, 880, 8381]
# 3	1800001318	PR	5	4	[898, 5602, 5053, 8877, 5282]	106	[9153, 3036, 898, 6687, 8381]	[9153, 6687, 8381, 8381, 8381]	[8381, 8381, 8381, 8381, 8381]	0	16	[5687, 8381, 8381, 8381, 8381]	1	[9593, 542, 4797]
# 4	1800001324	PR	13	4	[8381, 8381, 8381, 8381, 8381]	145	[8381, 8381, 8381, 8381, 8381]	[8381, 8381, 8381, 8381, 8381]	[7053, 6272, 8381, 8381, 8381]	2	17	[7759, 8381, 8381, 8381, 8381]	1	[542, 4530, 4797]

# 08 树模型训练与评估
import yaml
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def prepare_features(df_processed, processed_features, max_list_length=5):
    """展开列表特征"""
    df_tree = df_processed[processed_features].copy()
    
    for feat in processed_features:
        if isinstance(df_tree[feat].iloc[0], list):
            expanded = df_tree[feat].apply(pd.Series).iloc[:, :max_list_length]
            expanded.columns = [f"{feat}_{i}" for i in range(expanded.shape[1])]
            df_tree = df_tree.drop(columns=[feat]).join(expanded)
    
    return df_tree

def train_model(X, y, train_params):
    """训练LightGBM模型"""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)
    
    model = lgb.train(
        train_params,
        train_data,
        num_boost_round=train_params.pop('num_iterations', 1000),
        callbacks=[lgb.early_stopping(train_params.pop('early_stopping_rounds', 100))],
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid']
    )
    
    return model, X_train, X_val, y_train, y_val

def evaluate_model(model, X_train, X_val, y_train, y_val):
    """评估模型性能"""
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)
    
    print(f"训练集 AUC: {train_auc:.4f}")
    print(f"验证集 AUC: {val_auc:.4f}")
    
    return pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

# 树模型使用config.yml配置
with open('config/config.yml', 'r', encoding='utf-8') as f:
    tree_config = yaml.safe_load(f)

if 'log_type' in df_processed.columns:
    # 准备数据
    df_processed['label'] = df_processed['log_type'].apply(lambda x: 1 if x == 'PC' else 0)
    X = prepare_features(df_processed, processed_features)
    y = df_processed['label']
    
    # 训练模型
    train_params = {**tree_config['train'], 'verbose': -1, 'n_jobs': -1, 'seed': 42}
    model, X_train, X_val, y_train, y_val = train_model(X, y, train_params)
    
    # 评估并输出结果
    feature_importance = evaluate_model(model, X_train, X_val, y_train, y_val)
    print("\n特征重要性 (Top 20):")
    print(feature_importance.head(20))
else:
    print("错误: 找不到 'log_type' 列，无法进行模型训练。")

# 09 深度模型全流程：类似Huggig Face的框架 这里用的是tensorflow的框架
# 原项目深度模型的核心特点：
# 1. 使用 tf.data.Dataset 进行数据流处理
# 2. 通过 FeaturePipelineBuilder 构建特征处理管道
# 3. 支持 sparse、varlen_sparse、dense 三种特征类型
# 4. 使用 Embedding + BatchNorm + Dense + Dropout 的经典架构

import tensorflow as tf
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Concatenate
from keras.models import Model
from keras.regularizers import l2

# ---------------------------------------------------------------------------- #
# 1. 基于原项目架构的特征处理器
#    参考 src/models/deep/feature_pipeline.py
# ---------------------------------------------------------------------------- #

class FeaturePipelineBuilder:
    """特征处理管道构建器 - 基于原项目实现"""
    
    def __init__(self, feat_configs: list, verbose: bool = True):
        self.feat_configs = feat_configs
        self.verbose = verbose
        self.embedding_layers = {}
        self.pooling_layers = {}
    
    def build_feature_pipelines(self) -> list:
        """构建特征处理管道"""
        pipelines = []
        
        for config in self.feat_configs:
            feat_name = config.get('feat_name')
            feat_type = config.get('feat_type')
            
            if not feat_name or not feat_type:
                continue
                
            # 创建处理器序列
            processors = self._create_processors(config)
            if processors:
                pipelines.append((feat_name, processors))
        
        if self.verbose:
            print(f"成功构建 {len(pipelines)} 个特征处理管道")
            for feat_name, processors in pipelines:
                processor_names = [p.__class__.__name__ for p in processors]
                print(f"  {feat_name}: {' -> '.join(processor_names)}")
        
        return pipelines
    
    def _create_processors(self, config: dict) -> list:
        """根据特征类型创建处理器"""
        feat_name = config['feat_name']
        feat_type = config['feat_type']
        vocab_size = config.get('vocabulary_size', 1000)
        embed_dim = config.get('embedding_dim', 8)
        
        if feat_type == 'sparse':
            embedding = Embedding(
                input_dim=vocab_size,
                output_dim=embed_dim,
                name=f'{feat_name}_embedding',
                mask_zero=False
            )
            return [embedding]
        
        elif feat_type == 'varlen_sparse':
            embedding = Embedding(
                input_dim=vocab_size,
                output_dim=embed_dim,
                name=f'{feat_name}_embedding',
                mask_zero=True  # 变长特征需要masking
            )
            pooling = GlobalAveragePooling1D(name=f'{feat_name}_pooling')
            return [embedding, pooling]
        
        elif feat_type == 'dense':
            # 密集特征直接通过，可以加BN
            identity = tf.keras.layers.Lambda(lambda x: x, name=f'{feat_name}_identity')
            return [identity]
        
        return []

def process_feature_batch(features_dict: dict, pipelines: list) -> list:
    """处理特征批次数据"""
    outputs = []
    
    for feat_name, processors in pipelines:
        if feat_name not in features_dict:
            continue
        
        # 依次应用处理器
        feature_input = features_dict[feat_name]
        for processor in processors:
            feature_input = processor(feature_input)
        
        outputs.append(feature_input)
    
    return outputs

# ---------------------------------------------------------------------------- #
# 2. MLP 模型定义（基于原项目架构）
#    参考 src/models/deep/mlp.py
# ---------------------------------------------------------------------------- #

class DeepMLP(tf.keras.Model):
    """深度MLP模型 - 基于原项目实现"""
    
    def __init__(self, feat_configs: list, train_config: dict = None, verbose: bool = True):
        super(DeepMLP, self).__init__()
        
        # 构建特征处理管道
        pipeline_builder = FeaturePipelineBuilder(feat_configs, verbose=verbose)
        self.feature_pipelines = pipeline_builder.build_feature_pipelines()
        
        # 特征连接层
        self.concat_layer = Concatenate(axis=1)
        
        # 获取模型参数 - 从train.yml动态读取
        model_params = (train_config or {}).get('model', {})
        hidden_layers = model_params.get('layers', [128, 64, 32])  # 使用train.yml的默认值
        dropout_rates = model_params.get('dropout_rates', [0.3, 0.3, 0.2])  # 使用train.yml的默认值
        l2_reg = model_params.get('l2_regularization', 0.001)
        
        # 构建分类器
        self.classifier = self._build_classifier(hidden_layers, dropout_rates, l2_reg)
    
    def _build_classifier(self, hidden_layers: list, dropout_rates: list, l2_reg: float):
        """构建分类器网络"""
        layers = []
        
        # 添加BatchNorm
        layers.append(BatchNormalization())
        
        # 添加隐藏层
        for i, units in enumerate(hidden_layers):
            layers.append(Dense(
                units, 
                activation='relu',
                kernel_regularizer=l2(l2_reg)
            ))
            if i < len(dropout_rates):
                layers.append(Dropout(dropout_rates[i]))
        
        # 输出层
        layers.append(Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_reg)))
        
        return tf.keras.Sequential(layers)
    
    def call(self, features, training=None):
        """前向传播"""
        # 处理所有特征
        processed_outputs = process_feature_batch(features, self.feature_pipelines)
        
        if not processed_outputs:
            raise ValueError("没有有效的特征输出")
        
        # 合并特征
        if len(processed_outputs) > 1:
            concat_features = self.concat_layer(processed_outputs)
        else:
            concat_features = processed_outputs[0]
        
        # 应用分类器
        predictions = self.classifier(concat_features, training=training)
        return predictions

# ---------------------------------------------------------------------------- #
# 3. 数据准备函数（基于原项目的数据格式）
# ---------------------------------------------------------------------------- #

def prepare_tf_dataset_for_deep_model(df: pd.DataFrame, feat_config: dict, batch_size: int = 512) -> tf.data.Dataset:
    """准备TensorFlow数据集用于深度模型训练"""
    
    # 获取需要的特征 - 适配不同的配置文件格式
    if 'pipelines' in feat_config:
        # feat.yml 格式
        pipelines = feat_config['pipelines']
    elif 'process' in feat_config and 'pipelines' in feat_config['process']:
        # config.yml 格式
        pipelines = feat_config['process']['pipelines']
    else:
        raise ValueError("无法找到特征配置中的 pipelines")
    
    pipeline_feats = {p['feat_name']: p for p in pipelines}
    features_dict = {}
    
    for feat_name, config in pipeline_feats.items():
        if feat_name not in df.columns:
            continue
        
        feat_type = config.get('feat_type', 'sparse')
        
        if feat_type == 'sparse':
            # 单值特征，shape为 (batch_size,) - 这是关键！
            values = df[feat_name].values.astype(np.int32)
            features_dict[feat_name] = values
            
        elif feat_type == 'varlen_sparse':
            # 变长特征，需要padding
            sequences = df[feat_name].tolist()
            max_len = max(len(seq) if isinstance(seq, list) else 1 for seq in sequences)
            
            padded_sequences = []
            for seq in sequences:
                if isinstance(seq, list):
                    padded = seq + [0] * (max_len - len(seq))
                else:
                    padded = [int(seq)] + [0] * (max_len - 1)
                padded_sequences.append(padded[:max_len])
            
            features_dict[feat_name] = np.array(padded_sequences, dtype=np.int32)
            
        elif feat_type == 'dense':
            # 密集特征
            values = df[feat_name].values.astype(np.float32).reshape(-1, 1)
            features_dict[feat_name] = values
    
    # 准备标签
    labels = df['label'].values.astype(np.int32)
    
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((features_dict, labels))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

# ---------------------------------------------------------------------------- #
# 4. 训练流程
# ---------------------------------------------------------------------------- #

print("\n--- 开始原项目风格的深度模型训练 ---")

# 深度模型读取train.yml配置文件
print("🔄 读取深度模型配置文件 (train.yml)...")
with open('config/train.yml', 'r', encoding='utf-8') as f:
    train_config = yaml.safe_load(f)

# 显示读取的配置信息
print(f"📋 使用的训练配置:")
training_config = train_config.get('training', {})
print(f"  Batch Size: {training_config.get('batch_size', 'N/A')}")
print(f"  Epochs: {training_config.get('epochs', 'N/A')}")
print(f"  Learning Rate: {training_config.get('lr', 'N/A')}")
print(f"  Validation Split: {training_config.get('validation_split', 'N/A')}")

model_config = train_config.get('model', {})
print(f"  Model Layers: {model_config.get('layers', 'N/A')}")
print(f"  Dropout Rates: {model_config.get('dropout_rates', 'N/A')}")
print(f"  L2 Regularization: {model_config.get('l2_regularization', 'N/A')}")

# 确保有标签 - 为原始数据添加标签
if 'label' not in df_raw.columns:
    df_raw['label'] = df_raw['log_type'].apply(lambda x: 1 if x == 'PC' else 0)

# 1. 准备数据集 - 深度模型应该独立处理原始数据
print("🔄 重新处理原始数据用于深度模型...")
print("📋 原项目架构说明:")
print("  🌳 树模型: config.yml (特征工程) + config.yml (训练参数)")
print("  🧠 深度模型: feat.yml (特征工程) + train.yml (训练参数)")
print("  📊 两个模型独立处理相同的原始数据，各自优化")

# 重新从原始数据开始，应用feat.yml配置进行特征工程
df_deep_processed, _ = process_feature_pipelines(df_raw, deep_feat_config)

batch_size = train_config.get('training', {}).get('batch_size', 256)  # 使用train.yml的默认值
full_dataset = prepare_tf_dataset_for_deep_model(df_deep_processed, deep_feat_config, batch_size)

print(f"已创建TensorFlow数据集，batch_size={batch_size}")

# 验证数据集格式
for features, labels in full_dataset.take(1):
    print("数据集格式验证:")
    print(f"  特征数量: {len(features)}")
    for name, tensor in features.items():
        print(f"  - {name}: shape={tensor.shape}, dtype={tensor.dtype}")
    print(f"  标签: shape={labels.shape}, dtype={labels.dtype}")
    break

# 2. 创建模型
print("\n构建深度MLP模型...")
# 获取深度模型的特征配置
if 'pipelines' in deep_feat_config:
    deep_pipelines = deep_feat_config['pipelines']
elif 'process' in deep_feat_config and 'pipelines' in deep_feat_config['process']:
    deep_pipelines = deep_feat_config['process']['pipelines']
else:
    raise ValueError("无法找到深度模型特征配置中的 pipelines")

deep_model = DeepMLP(deep_pipelines, train_config, verbose=True)

# 3. 编译模型
learning_rate = train_config.get('training', {}).get('lr', 0.0005)  # 使用train.yml的默认值
deep_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy')
    ]
)

print(f"模型编译完成，学习率: {learning_rate}")

# 4. 数据分割（使用配置文件中的validation_split）
validation_split = train_config.get('training', {}).get('validation_split', 0.2)
train_size = int((1 - validation_split) * len(df_deep_processed))

print(f"\n📊 数据使用情况:")
print(f"  总数据量: {len(df_deep_processed):,}")
print(f"  训练集: {train_size:,} ({(1-validation_split)*100:.1f}%)")
print(f"  验证集: {len(df_deep_processed)-train_size:,} ({validation_split*100:.1f}%)")
print(f"  训练批次数: {train_size // batch_size}")
print(f"  验证批次数: {(len(df_deep_processed)-train_size) // batch_size}")

train_dataset = full_dataset.take(train_size // batch_size)
val_dataset = full_dataset.skip(train_size // batch_size)

# 5. 训练模型
epochs = train_config.get('training', {}).get('epochs', 2)  # 使用train.yml的默认值
print(f"\n开始训练深度模型... (epochs={epochs})")

history = deep_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    verbose=1
)

# 6. 输出结果
print("\n--- 深度模型训练完成 ---")
if 'val_auc' in history.history:
    final_train_auc = history.history['auc'][-1]
    final_val_auc = history.history['val_auc'][-1]
    print(f"最终训练集 AUC: {final_train_auc:.4f}")
    print(f"最终验证集 AUC: {final_val_auc:.4f}")
    print(f"AUC差异: {final_train_auc - final_val_auc:.4f}")
else:
    final_auc = history.history['auc'][-1]
    print(f"最终 AUC: {final_auc:.4f}")

print(f"\n📋 深度模型架构总结:")
print(f"  🔧 特征处理管道: {len(deep_model.feature_pipelines)} 个")
print(f"  🏗️  分类器层数: {len([l for l in deep_model.classifier.layers if isinstance(l, Dense)])}")
print(f"  📊 总参数量: {deep_model.count_params():,}")

# 显示深度模型训练结果
print(f"\n🔄 深度模型训练结果:")
if 'val_auc' in history.history:
    print(f"  验证集AUC: {final_val_auc:.4f}")
    print(f"  训练集AUC: {history.history['auc'][-1]:.4f}")
else:
    print(f"  AUC: {final_auc:.4f}")

# 如果需要和树模型对比，请重新运行完整的流程

print("\n✅ 完整的特征工程 -> 树模型 -> 深度模型流水线已完成！")

# 10AFM模型训练

# ---------------------------------------------------------------------------- #
# 5. AFM模型实现 (Attentional Factorization Machines)
# ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
from torch import Tensor
from itertools import combinations
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import numpy as np

class AttentionalFactorizationMachine(nn.Module):
    """
    注意力因子分解机 (AFM)
    
    参数:
    - num_fields: int. 特征字段数量
    - emb_dim: int. 嵌入维度
    - attn_dim: int. 注意力维度
    - num_classes: int. 类别数量. 默认: 1
    - bias: bool. 是否使用偏置. 默认: True
    - dropout_rate: float. Dropout率. 默认: 0.2
    
    输入:
    - x: Tensor. 形状: (batch_size, num_fields, emb_dim)
    
    返回:
    - y: Tensor. 形状: (batch_size, num_classes)
    """

    def __init__(
        self,
        num_fields: int,
        emb_dim: int,
        attn_dim: int,
        num_classes: int = 1,
        bias: bool = True,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        
        # 特征字段组合
        combs = list(combinations(range(num_fields), 2))
        self.comb_i = [c[0] for c in combs]
        self.comb_j = [c[1] for c in combs]
        
        # 注意力网络
        self.linear_attn = nn.Sequential(
            nn.Linear(emb_dim, attn_dim, bias),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(attn_dim, 1, bias=False),
        )
        
        # 输出层
        self.fc = nn.Linear(emb_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Arguments:
            x -- Tensor. shape: (batch_size, num_fields, emb_dim)
        
        Returns:
            y -- Tensor. shape: (batch_size, num_classes)
        """
        print(f"    🔍 AFM前向传播开始，输入形状: {x.shape}")
        
        # 获取特征交互对
        print(f"    📊 计算特征交互对，组合数: {len(self.comb_i)}")
        x_i = x[:, self.comb_i]  # (batch_size, num_pairs, emb_dim)
        x_j = x[:, self.comb_j]  # (batch_size, num_pairs, emb_dim)
        print(f"    ✅ 特征交互对计算完成，x_i: {x_i.shape}, x_j: {x_j.shape}")
        
        # 元素级乘积得到交互特征
        x_cross = x_i * x_j  # (batch_size, num_pairs, emb_dim)
        print(f"    ✅ 交互特征计算完成，x_cross: {x_cross.shape}")
        
        # 计算注意力分数
        print(f"    🧠 开始计算注意力分数...")
        attn_score = self.linear_attn(x_cross)  # (batch_size, num_pairs, 1)
        print(f"    ✅ 注意力分数计算完成，attn_score: {attn_score.shape}")
        
        attn_score = torch.softmax(attn_score, dim=1)  # 注意力权重归一化
        print(f"    ✅ 注意力权重归一化完成")
        
        # 加权求和得到最终特征表示
        f = torch.sum(attn_score * x_cross, dim=1)  # (batch_size, emb_dim)
        print(f"    ✅ 加权求和完成，f: {f.shape}")
        
        # 应用dropout
        f = self.dropout(f)
        print(f"    ✅ Dropout完成")
        
        # 输出预测
        y = self.fc(f)  # (batch_size, num_classes)
        print(f"    ✅ 最终输出完成，y: {y.shape}")
        
        return y


class EmbeddingLayer(nn.Module):
    """
    嵌入层 - 处理稀疏特征和变长稀疏特征
    """
    
    def __init__(self, feat_configs: list, verbose: bool = True):
        super().__init__()
        self.feat_configs = feat_configs
        self.embeddings = nn.ModuleDict()
        self.feat_info = {}
        
        if verbose:
            print("🔧 构建嵌入层...")
        
        for config in feat_configs:
            feat_name = config['feat_name']
            feat_type = config['feat_type']
            vocab_size = config['vocabulary_size']
            emb_dim = config['embedding_dim']
            
            # 创建嵌入层
            self.embeddings[feat_name] = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
            
            # 存储特征信息
            self.feat_info[feat_name] = {
                'type': feat_type,
                'vocab_size': vocab_size,
                'emb_dim': emb_dim
            }
            
            if verbose:
                print(f"  - {feat_name}: {feat_type}, vocab={vocab_size}, emb_dim={emb_dim}")
        
        # 初始化嵌入权重
        self._init_embeddings()
    
    def _init_embeddings(self):
        """初始化嵌入权重"""
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
            # padding idx设为0
            with torch.no_grad():
                emb.weight[0].fill_(0)
    
    def forward(self, features_dict):
        """
        前向传播
        
        Args:
            features_dict: dict, 特征字典 {feat_name: tensor}
        
        Returns:
            embedded_features: list of tensors, 每个tensor形状为 (batch_size, emb_dim)
        """
        print(f"  🔧 嵌入层开始处理 {len(features_dict)} 个特征...")
        embedded_features = []
        
        for i, config in enumerate(self.feat_configs):
            feat_name = config['feat_name']
            feat_type = config['feat_type']
            
            if feat_name not in features_dict:
                print(f"    ⚠️  特征 {feat_name} 不在输入中，跳过")
                continue
            
            feat_tensor = features_dict[feat_name]
            print(f"    📊 处理特征 {i+1}/{len(self.feat_configs)}: {feat_name} ({feat_type}), 输入形状: {feat_tensor.shape}")
            
            if feat_type == 'sparse':
                # 稀疏特征: (batch_size,) -> (batch_size, emb_dim)
                emb = self.embeddings[feat_name](feat_tensor)
                print(f"      ✅ 稀疏特征嵌入完成，输出形状: {emb.shape}")
                embedded_features.append(emb)
                
            elif feat_type == 'varlen_sparse':
                # 变长稀疏特征: (batch_size, max_len) -> (batch_size, emb_dim)
                emb = self.embeddings[feat_name](feat_tensor)  # (batch_size, max_len, emb_dim)
                print(f"      📊 变长特征嵌入: {emb.shape}")
                
                # 平均池化
                mask = (feat_tensor != 0).float().unsqueeze(-1)  # (batch_size, max_len, 1)
                print(f"      📊 掩码形状: {mask.shape}")
                
                emb_masked = emb * mask  # (batch_size, max_len, emb_dim)
                emb_pooled = emb_masked.sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (batch_size, emb_dim)
                print(f"      ✅ 变长特征池化完成，输出形状: {emb_pooled.shape}")
                embedded_features.append(emb_pooled)
        
        print(f"  ✅ 嵌入层处理完成，生成 {len(embedded_features)} 个特征嵌入")
        return embedded_features


class PushAFMModel(nn.Module):
    """
    完整的AFM模型 - 专门用于推送分类任务
    """
    
    def __init__(self, feat_configs: list, train_config: dict = None, verbose: bool = True):
        super().__init__()
        
        self.feat_configs = feat_configs
        self.train_config = train_config or {}
        
        # 获取嵌入维度
        emb_dims = [config['embedding_dim'] for config in feat_configs]
        if len(set(emb_dims)) != 1:
            raise ValueError("所有特征的嵌入维度必须相同")
        self.emb_dim = emb_dims[0]
        
        # 特征数量
        self.num_fields = len(feat_configs)
        
        if verbose:
            print(f"🏗️  构建AFM模型...")
            print(f"  特征字段数: {self.num_fields}")
            print(f"  嵌入维度: {self.emb_dim}")
        
        # 嵌入层
        self.embedding_layer = EmbeddingLayer(feat_configs, verbose=verbose)
        
        # AFM参数
        model_config = self.train_config.get('model', {})
        attn_dim = model_config.get('attention_dim', 64)  # 注意力维度
        dropout_rate = model_config.get('dropout_rate', 0.2)
        
        if verbose:
            print(f"  注意力维度: {attn_dim}")
            print(f"  Dropout率: {dropout_rate}")
        
        # AFM核心
        self.afm = AttentionalFactorizationMachine(
            num_fields=self.num_fields,
            emb_dim=self.emb_dim,
            attn_dim=attn_dim,
            num_classes=1,
            dropout_rate=dropout_rate
        )
        
        # Sigmoid激活函数用于二分类
        self.sigmoid = nn.Sigmoid()
        
        if verbose:
            total_params = sum(p.numel() for p in self.parameters())
            print(f"  总参数量: {total_params:,}")
    
    def forward(self, features_dict):
        """
        前向传播
        """
        # 1. 嵌入层处理
        embedded_features = self.embedding_layer(features_dict)  # list of (batch_size, emb_dim)
        
        # 2. 堆叠成AFM输入格式
        x = torch.stack(embedded_features, dim=1)  # (batch_size, num_fields, emb_dim)
        
        # 3. AFM计算
        logits = self.afm(x)  # (batch_size, 1)
        
        # 4. Sigmoid激活
        probs = self.sigmoid(logits)  # (batch_size, 1)
        
        return probs.squeeze(-1)  # (batch_size,)


class PushDataset(Dataset):
    """
    推送数据的PyTorch Dataset
    """
    
    def __init__(self, features_dict: dict, labels: np.ndarray):
        print(f"  🔧 开始创建PushDataset...")
        print(f"    📊 输入特征数量: {len(features_dict)}")
        print(f"    📊 标签数量: {len(labels)}")
        
        self.features_dict = features_dict
        self.labels = labels
        self.length = len(labels)
        
        # 转换为tensor
        print(f"  🔄 开始转换特征为tensor...")
        self.features_tensor = {}
        
        for i, (name, values) in enumerate(features_dict.items()):
            print(f"    📊 转换特征 {i+1}/{len(features_dict)}: {name}")
            print(f"      输入类型: {type(values)}, 形状: {values.shape}, dtype: {values.dtype}")
            
            try:
                if isinstance(values, np.ndarray):
                    if values.dtype in [np.int32, np.int64]:
                        tensor = torch.from_numpy(values).long()
                    elif values.dtype in [np.float32, np.float64]:
                        tensor = torch.from_numpy(values.astype(np.int64)).long()
                    else:
                        tensor = torch.from_numpy(values.astype(np.int64)).long()
                else:
                    tensor = torch.tensor(values, dtype=torch.long)
                
                self.features_tensor[name] = tensor
                print(f"      ✅ 转换成功，tensor形状: {tensor.shape}, dtype: {tensor.dtype}")
                
            except Exception as e:
                print(f"      ❌ 转换失败: {e}")
                # 尝试备用方案
                try:
                    if isinstance(values, np.ndarray):
                        tensor = torch.from_numpy(values.astype(np.int64)).long()
                    else:
                        tensor = torch.tensor(values, dtype=torch.long)
                    self.features_tensor[name] = tensor
                    print(f"      ✅ 备用转换成功，tensor形状: {tensor.shape}")
                except Exception as e2:
                    print(f"      ❌ 备用转换也失败: {e2}")
                    raise e2
        
        print(f"  🔄 转换标签为tensor...")
        try:
            self.labels_tensor = torch.from_numpy(labels).float()
            print(f"    ✅ 标签转换成功，形状: {self.labels_tensor.shape}")
        except Exception as e:
            print(f"    ❌ 标签转换失败: {e}")
            raise e
        
        print(f"  ✅ PushDataset创建完成！")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        features = {name: tensor[idx] for name, tensor in self.features_tensor.items()}
        label = self.labels_tensor[idx]
        return features, label


def prepare_afm_dataset(df: pd.DataFrame, feat_config: dict, batch_size: int = 256) -> tuple:
    """
    准备AFM训练数据
    """
    # 获取特征配置
    if 'pipelines' in feat_config:
        pipelines = feat_config['pipelines']
    elif 'process' in feat_config and 'pipelines' in feat_config['process']:
        pipelines = feat_config['process']['pipelines']
    else:
        raise ValueError("无法找到特征配置中的 pipelines")
    
    pipeline_feats = {p['feat_name']: p for p in pipelines}
    features_dict = {}
    
    for feat_name, config in pipeline_feats.items():
        if feat_name not in df.columns:
            continue
        
        feat_type = config.get('feat_type', 'sparse')
        
        if feat_type == 'sparse':
            # 单值特征
            values = df[feat_name].values.astype(np.int32)
            features_dict[feat_name] = values
            
        elif feat_type == 'varlen_sparse':
            # 变长特征，需要padding
            sequences = df[feat_name].tolist()
            max_len = max(len(seq) if isinstance(seq, list) else 1 for seq in sequences)
            
            padded_sequences = []
            for seq in sequences:
                if isinstance(seq, list):
                    padded = seq + [0] * (max_len - len(seq))
                else:
                    padded = [int(seq)] + [0] * (max_len - 1)
                padded_sequences.append(padded[:max_len])
            
            features_dict[feat_name] = np.array(padded_sequences, dtype=np.int32)
    
    # 准备标签
    labels = df['label'].values.astype(np.float32)
    
    return features_dict, labels


def train_afm_model(model, train_loader, val_loader, train_config: dict, device):
    """
    训练AFM模型
    """
    # 优化器配置
    training_config = train_config.get('training', {})
    lr = training_config.get('lr', 0.001)
    weight_decay = training_config.get('weight_decay', 0.001)
    epochs = training_config.get('epochs', 2)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    print(f"\n🚀 开始AFM模型训练...")
    print(f"  学习率: {lr}")
    print(f"  权重衰减: {weight_decay}")
    print(f"  训练轮数: {epochs}")
    print(f"  设备: {device}")
    
    # 添加调试信息
    print(f"\n🔧 训练数据检查:")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    # 测试第一个batch以检查数据格式
    print(f"\n🧪 测试第一个batch...")
    try:
        first_batch = next(iter(train_loader))
        features, labels = first_batch
        print(f"  ✅ 数据加载成功")
        print(f"  特征数量: {len(features)}")
        print(f"  标签形状: {labels.shape}")
        
        # 测试模型前向传播
        print(f"  🧠 测试模型前向传播...")
        model.eval()
        with torch.no_grad():
            # 移动到设备
            features_device = {name: tensor.to(device) for name, tensor in features.items()}
            labels_device = labels.to(device)
            
            start_time = time.time()
            predictions = model(features_device)
            forward_time = time.time() - start_time
            
            print(f"  ✅ 前向传播成功，耗时: {forward_time:.2f}秒")
            print(f"  预测形状: {predictions.shape}")
            print(f"  预测范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
            
    except Exception as e:
        print(f"  ❌ 数据或模型测试失败: {e}")
        return None
    
    best_val_auc = 0.0
    train_losses = []
    val_aucs = []
    
    for epoch in range(epochs):
        print(f"\n🔄 开始第 {epoch+1}/{epochs} 轮训练...")
        
        # 训练阶段
        model.train()
        total_loss = 0.0
        train_predictions = []
        train_targets = []
        
        # 添加时间统计
        epoch_start_time = time.time()
        batch_times = []
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            
            # 移动到设备
            features = {name: tensor.to(device) for name, tensor in features.items()}
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_predictions.extend(predictions.detach().cpu().numpy())
            train_targets.extend(labels.detach().cpu().numpy())
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # 更频繁的进度输出
            if batch_idx % 10 == 0:  # 每10个batch输出一次
                avg_batch_time = np.mean(batch_times[-10:])  # 最近10个batch的平均时间
                eta = avg_batch_time * (len(train_loader) - batch_idx - 1)  # 预估剩余时间
                print(f"    Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, "
                      f"Time: {batch_time:.2f}s, ETA: {eta:.1f}s")
        
        # 计算训练指标
        train_auc = roc_auc_score(train_targets, train_predictions)
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        epoch_time = time.time() - epoch_start_time
        
        print(f"\n⏱️  第{epoch+1}轮训练完成，耗时: {epoch_time:.1f}秒")
        print(f"    平均batch时间: {np.mean(batch_times):.2f}秒")
        
        # 验证阶段
        print(f"🔍 开始验证...")
        model.eval()
        val_predictions = []
        val_targets = []
        val_loss = 0.0
        
        val_start_time = time.time()
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(val_loader):
                features = {name: tensor.to(device) for name, tensor in features.items()}
                labels = labels.to(device)
                
                predictions = model(features)
                loss = criterion(predictions, labels)
                
                val_loss += loss.item()
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
                if batch_idx % 20 == 0:
                    print(f"    验证Batch {batch_idx}/{len(val_loader)}")
        
        val_time = time.time() - val_start_time
        print(f"✅ 验证完成，耗时: {val_time:.1f}秒")
        
        # 计算验证指标
        val_auc = roc_auc_score(val_targets, val_predictions)
        avg_val_loss = val_loss / len(val_loader)
        val_aucs.append(val_auc)
        
        print(f"\n📊 Epoch {epoch+1}/{epochs} 结果:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  训练AUC: {train_auc:.4f}")
        print(f"  验证损失: {avg_val_loss:.4f}")
        print(f"  验证AUC: {val_auc:.4f}")
        print(f"  总耗时: {epoch_time + val_time:.1f}秒")
        
        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            print(f"  🎉 新的最佳验证AUC: {best_val_auc:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_aucs': val_aucs,
        'best_val_auc': best_val_auc
    }


# ---------------------------------------------------------------------------- #
# 6. AFM模型训练主流程
# ---------------------------------------------------------------------------- #

print("\n" + "="*80)
print("🔥 开始AFM模型训练")
print("="*80)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  使用设备: {device}")

# 确保有标签
if 'label' not in df_raw.columns:
    df_raw['label'] = df_raw['log_type'].apply(lambda x: 1 if x == 'PC' else 0)

# 1. 重新处理数据 (使用feat.yml配置)
print("\n🔄 为AFM模型处理数据...")
df_afm_processed, _ = process_feature_pipelines(df_raw, deep_feat_config)

print(f"📊 AFM数据统计:")
print(f"  总样本数: {len(df_afm_processed):,}")
print(f"  正样本数: {df_afm_processed['label'].sum():,}")
print(f"  负样本数: {(df_afm_processed['label'] == 0).sum():,}")
print(f"  正样本比例: {df_afm_processed['label'].mean():.3f}")

# 2. 准备AFM数据集
print("\n🔧 准备AFM数据集...")
features_dict, labels = prepare_afm_dataset(df_afm_processed, deep_feat_config)

print(f"特征字段: {list(features_dict.keys())}")
for name, values in features_dict.items():
    print(f"  - {name}: shape={values.shape}, dtype={values.dtype}")

# 3. 创建训练/验证分割
validation_split = train_config.get('training', {}).get('validation_split', 0.2)
train_size = int((1 - validation_split) * len(labels))

print(f"\n📊 数据分割:")
print(f"  训练集: {train_size:,} ({(1-validation_split)*100:.1f}%)")
print(f"  验证集: {len(labels)-train_size:,} ({validation_split*100:.1f}%)")

# 分割数据
train_features = {name: values[:train_size] for name, values in features_dict.items()}
val_features = {name: values[train_size:] for name, values in features_dict.items()}
train_labels = labels[:train_size]
val_labels = labels[train_size:]

# 4. 创建PyTorch数据集和数据加载器
batch_size = train_config.get('training', {}).get('batch_size', 256)

# 如果是CPU训练，减少batch_size以提高速度
if device.type == 'cpu':
    batch_size = min(batch_size, 16)  # CPU上使用更小的batch_size
    print(f"⚠️  检测到CPU训练，减少batch_size到: {batch_size}")

print(f"\n📦 创建数据集...")
train_dataset = PushDataset(train_features, train_labels)
val_dataset = PushDataset(val_features, val_labels)

print(f"  训练数据集大小: {len(train_dataset):,}")
print(f"  验证数据集大小: {len(val_dataset):,}")

# 测试数据集
print(f"🧪 测试数据集...")
try:
    sample_features, sample_label = train_dataset[0]
    print(f"  ✅ 数据集创建成功")
    print(f"  样本特征数量: {len(sample_features)}")
    print(f"  样本标签: {sample_label}")
except Exception as e:
    print(f"  ❌ 数据集测试失败: {e}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"  训练批次数: {len(train_loader)}")
print(f"  验证批次数: {len(val_loader)}")
print(f"  使用batch_size: {batch_size}")

# 测试数据加载器
print(f"🧪 测试数据加载器...")
try:
    start_time = time.time()
    first_batch = next(iter(train_loader))
    load_time = time.time() - start_time
    print(f"  ✅ 数据加载成功，耗时: {load_time:.2f}秒")
except Exception as e:
    print(f"  ❌ 数据加载失败: {e}")

# 5. 创建AFM模型
print("\n🏗️  构建AFM模型...")

# 获取特征配置
if 'pipelines' in deep_feat_config:
    afm_pipelines = deep_feat_config['pipelines']
elif 'process' in deep_feat_config and 'pipelines' in deep_feat_config['process']:
    afm_pipelines = deep_feat_config['process']['pipelines']
else:
    raise ValueError("无法找到AFM特征配置中的 pipelines")

# 添加AFM特定配置到train_config
afm_train_config = train_config.copy()
if 'model' not in afm_train_config:
    afm_train_config['model'] = {}
afm_train_config['model']['attention_dim'] = 32  # AFM注意力维度（减少计算量）
afm_train_config['model']['dropout_rate'] = 0.2  # AFM的dropout率

# 创建模型
afm_model = PushAFMModel(afm_pipelines, afm_train_config, verbose=True)
afm_model.to(device)

print(f"\n📋 AFM模型架构:")
print(f"  🔧 特征字段数: {afm_model.num_fields}")
print(f"  📊 嵌入维度: {afm_model.emb_dim}")
print(f"  🧠 注意力维度: {afm_train_config['model']['attention_dim']}")
print(f"  💧 Dropout率: {afm_train_config['model']['dropout_rate']}")
print(f"  📊 总参数量: {sum(p.numel() for p in afm_model.parameters()):,}")

# 6. 训练模型
training_results = train_afm_model(afm_model, train_loader, val_loader, afm_train_config, device)

# 7. 输出最终结果
print("\n" + "="*80)
print("🎉 AFM模型训练完成！")
print("="*80)

print(f"\n📊 最终结果:")
print(f"  最佳验证AUC: {training_results['best_val_auc']:.4f}")
print(f"  最终训练损失: {training_results['train_losses'][-1]:.4f}")
print(f"  AUC提升曲线: {[f'{auc:.4f}' for auc in training_results['val_aucs']]}")

print(f"\n🔄 模型对比:")
print(f"  🌳 树模型(LightGBM): 基于config.yml特征工程")
print(f"  🧠 MLP模型: 基于feat.yml特征工程")
print(f"  🔥 AFM模型: 基于feat.yml特征工程 + 注意力机制")
print(f"     └─ 最佳验证AUC: {training_results['best_val_auc']:.4f}")

print(f"\n✅ AFM模型训练流水线完成！")
print(f"💡 AFM通过注意力机制自动学习特征交互的重要性，")
print(f"   相比传统FM模型能更好地处理特征交互！")
