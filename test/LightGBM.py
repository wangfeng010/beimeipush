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
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 添加项目路径
project_root = os.getcwd()
env_path = os.path.join(project_root, 'env')
if env_path not in sys.path:
    sys.path.insert(0, env_path)

# 02 根据配置文件加载原始数据
def load_raw_data_from_config():
    """根据data.yml配置文件加载原始数据，支持本地和线上环境"""
    
    # 加载数据配置
    with open('config/data.yml', 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 检测文件类型和环境
    train_dir = data_config['train_dir']
    
    # 检查CSV和TXT文件
    csv_files = glob(os.path.join(train_dir, '*.csv'))
    txt_files = glob(os.path.join(train_dir, '*.txt'))
    
    if csv_files:
        # 本地环境 - 使用CSV格式
        print("检测到CSV文件，使用本地环境配置")
        csv_config = data_config['csv_format']
        separator, header = csv_config['separator'], csv_config['header']
        
        print(f"分隔符: '{separator}', 表头行: {header}, 文件数量: {len(csv_files)}")
        
        # 读取CSV文件
        dfs = [pd.read_csv(f, sep=separator, header=header) for f in csv_files]
        
    elif txt_files:
        # 线上环境 - 使用TXT格式
        print("检测到TXT文件，使用线上环境配置")
        txt_config = data_config.get('txt_format', {'separator': '\t', 'header': None})
        separator, header = txt_config['separator'], txt_config['header']
        
        # 从列表中提取列名
        raw_columns = [list(item.keys())[0] for item in data_config['raw_data_columns']]
        
        print(f"分隔符: '{separator}', 表头行: {header}, 文件数量: {len(txt_files)}")
        print(f"预定义列名: {raw_columns}")
        
        # 读取TXT文件
        dfs = [pd.read_csv(f, sep=separator, header=header, names=raw_columns) for f in txt_files]
    else:
        raise ValueError(f"在目录 {train_dir} 中未找到CSV或TXT文件")
    
    df_raw = pd.concat(dfs, ignore_index=True)
    print(f"形状{df_raw.shape}, 列名: {list(df_raw.columns)}")
    
    return df_raw

# 加载原始数据
df_raw = load_raw_data_from_config()

# 显示数据样例（简化版）
print(df_raw.iloc[0].to_dict())

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

# 看一下一个pipeline中对一个特征的操作 他被解析成了什么结构
# 适配 config.yml 和 feat.yml 的不同结构
if 'pipelines' in feat_config:
    # feat.yml 格式
    example_pipeline = feat_config['pipelines'][0]
elif 'process' in feat_config and 'pipelines' in feat_config['process']:
    # config.yml 格式
    example_pipeline = feat_config['process']['pipelines'][0]
else:
    print("无法找到特征配置格式")
    example_pipeline = None

if example_pipeline:
    pprint(example_pipeline)

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

print("数据结构对比:")
print(f"原始列数: {len(df_raw.columns)}")
print(f"处理后列数: {len(df_processed.columns)}")
print(f"新增列数: {len(df_processed.columns) - len(df_raw.columns)}")

print("\n原始列名:")
print(list(df_raw.columns))

print("\n新增列名:")
new_columns = [col for col in df_processed.columns if col not in df_raw.columns]
print(new_columns)

# 查看成功生成的特征
print("成功生成的特征详情:")
for feat_name in processed_features:
    if feat_name in df_processed.columns:
        sample_data = df_processed[feat_name].iloc[0]
        data_type = type(sample_data).__name__
        print(f"  {feat_name}: {data_type} = {sample_data}")

print("最终处理结果预览:")
display_cols = ['user_id', 'log_type'] + processed_features
display_cols = [col for col in display_cols if col in df_processed.columns]

print(df_processed[display_cols].head())

def prepare_features(df_processed, processed_features, max_list_length=5):
    """展开列表特征 树模型的需求"""
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

