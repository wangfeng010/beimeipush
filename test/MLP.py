import tensorflow as tf
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Concatenate
from keras.models import Model
from keras.regularizers import l2
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

# 02 根据配置文件加载原始数据
# 和树模型部分一样
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

# 加载深度模型的训练和模型配置（train.yml）
with open('config/train.yml', 'r', encoding='utf-8') as f:
    train_config = yaml.safe_load(f)

training_config = train_config.get('training', {})
model_config = train_config.get('model', {})

# 加载深度模型的特征配置 (feat.yml)
with open('config/feat.yml', 'r', encoding='utf-8') as f:
    deep_feat_config = yaml.safe_load(f)

print(training_config)
print(model_config)
print(deep_feat_config)

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
        print("无法找到特征配置中的 pipelines")
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

# label的处理
if 'label' not in df_raw.columns:
    df_raw['label'] = df_raw['log_type'].apply(lambda x: 1 if x == 'PC' else 0)

# 2.4节 只有这里是和1.4节不一样的
df_deep_processed, processed_features = process_feature_pipelines(df_raw, deep_feat_config)

print("数据结构对比:")
print(f"原始列数: {len(df_raw.columns)}")
print(f"处理后列数: {len(df_deep_processed.columns)}")
print(f"新增列数: {len(df_deep_processed.columns) - len(df_raw.columns)}")

print("\n原始列名:")
print(list(df_raw.columns))

print("\n新增列名:")
new_columns = [col for col in df_deep_processed.columns if col not in df_raw.columns]
print(new_columns)

# 查看成功生成的特征
print("成功生成的特征详情:")
for feat_name in processed_features:
    if feat_name in df_deep_processed.columns:
        sample_data = df_deep_processed[feat_name].iloc[0]
        data_type = type(sample_data).__name__
        print(f"  {feat_name}: {data_type} = {sample_data}")

print("最终处理结果预览:")
display_cols = ['user_id', 'log_type'] + processed_features
display_cols = [col for col in display_cols if col in df_deep_processed.columns]

print(df_deep_processed[display_cols].head())

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

batch_size = train_config.get('training', {}).get('batch_size', 256)  # 使用train.yml的默认值
full_dataset = prepare_tf_dataset_for_deep_model(df_deep_processed, deep_feat_config, batch_size)

# 查看df数据集
for features, labels in full_dataset.take(1):
    print("数据集格式验证:")
    print(f"  特征数量: {len(features)}")
    for name, tensor in features.items():
        print(f"  - {name}: shape={tensor.shape}, dtype={tensor.dtype}")
    print(f"  标签: shape={labels.shape}, dtype={labels.dtype}")
    break

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

if 'pipelines' in deep_feat_config:
    deep_pipelines = deep_feat_config['pipelines']
elif 'process' in deep_feat_config and 'pipelines' in deep_feat_config['process']:
    deep_pipelines = deep_feat_config['process']['pipelines']
else:
    raise ValueError("无法找到深度模型特征配置中的 pipelines")

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

print(f"深度模型架构总结:")
print(f"特征处理管道: {len(deep_model.feature_pipelines)} 个")
print(f"分类器层数: {len([l for l in deep_model.classifier.layers if isinstance(l, Dense)])}")
print(f"总参数量: {deep_model.count_params():,}")

# 显示深度模型训练结果
print(f"深度模型训练结果:")
if 'val_auc' in history.history:
    print(f"验证集AUC: {final_val_auc:.4f}")
    print(f"训练集AUC: {history.history['auc'][-1]:.4f}")
else:
    print(f"AUC: {final_auc:.4f}")

