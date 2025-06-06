import lightgbm as lgb
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config.base_config import AppConfig, TrainsetConfig
from src.config.yaml_config import load_config_from_yaml
from src.preprocess.preprocess import run_one_op_pd
from src.utils.config_utils import _load_yaml_config, _determine_exclude_features


class PushClassifier:
    def __init__(
        self,
        config_dir: str,
        mode: str = "train",
        varlen_max: int = 5,
    ) -> None:
        self.cfg = self._get_config(config_dir)
        self.varlen_max = varlen_max
        self.model_config = self.cfg.train
        
        # 加载特征排除配置（从feat.yml读取排除设置）
        self.exclude_features = self._load_exclude_features()
        if self.exclude_features:
            logger.info(f"树模型将排除以下特征: {self.exclude_features}")
        else:
            logger.info("树模型使用所有特征（无排除）")

    def _get_config(self, config_dir: str) -> AppConfig:
        config = load_config_from_yaml(config_dir, AppConfig)
        logger.info(f"base config: \n{config}")
        return config

    def _load_exclude_features(self):
        """从feat.yml加载特征排除配置"""
        try:
            feat_config = _load_yaml_config("config/feat.yml")
            exclude_features = _determine_exclude_features(feat_config)
            return exclude_features
        except Exception as e:
            logger.warning(f"无法加载特征排除配置: {e}，将使用所有特征")
            return []

    def _should_exclude_feature(self, feature_name) -> bool:
        """判断特征是否应该被排除"""
        if not self.exclude_features:
            return False
            
        # 如果feature_name是列表，检查每个元素
        if isinstance(feature_name, list):
            for fname in feature_name:
                if self._should_exclude_feature(fname):
                    return True
            return False
        
        # 如果不是字符串，跳过
        if not isinstance(feature_name, str):
            return False
            
        for exclude_feature in self.exclude_features:
            # 精确匹配或前缀匹配（例如：user_propernoun_hash）
            if (feature_name == exclude_feature or 
                feature_name.startswith(exclude_feature + '_') or
                feature_name.endswith('_' + exclude_feature)):
                logger.info(f"树模型排除特征: {feature_name} (匹配规则: {exclude_feature})")
                return True
        return False

    def _data_preprocess(self, x: pd.DataFrame) -> pd.DataFrame:
        # 使用原始config.yml中的特征处理管道
        pipelines = (
            self.cfg.features.process.pipelines
            + self.cfg.features.interactions.pipelines
        )
        
        for pipe in pipelines:
            # 检查管道是否应该被排除
            pipe_name = getattr(pipe, 'feat_name', '')
            if self._should_exclude_feature(pipe_name):
                logger.info(f"树模型跳过特征管道: {pipe_name}")
                continue
                
            for op in pipe.operations:
                try:
                    # 检查操作的输入列是否应该被排除
                    col_in = getattr(op, 'col_in', '')
                    if col_in and self._should_exclude_feature(col_in):
                        logger.info(f"树模型跳过操作: {op} (输入列被排除: {col_in})")
                        continue
                        
                    x = run_one_op_pd(x, op)
                except Exception as e:
                    logger.debug(
                        f"input:\n{x.get(op.col_in, 'N/A')}. op:{op.col_in}. got error{e}"
                    )
                    raise e
        
        # label处理
        for pipe in self.cfg.features.label_process.pipelines:
            for op in pipe.operations:
                x = run_one_op_pd(x, op)
        
        # 过滤输出列，排除被排除的特征
        valid_feat_names = [
            feat_name for feat_name in self.cfg.features.feat_names 
            if not self._should_exclude_feature(feat_name)
        ]
        
        out_columns = valid_feat_names + self.cfg.datasets.trainset.label_columns
        
        logger.info(f"原始特征数: {len(self.cfg.features.feat_names)}")
        logger.info(f"过滤后特征数: {len(valid_feat_names)}")
        logger.info(f"排除的特征数: {len(self.cfg.features.feat_names) - len(valid_feat_names)}")
        
        return x[out_columns]

    def _feat_selection(self, x: pd.DataFrame, max_col_num: int) -> pd.DataFrame:
        # 过滤特征名称，排除被排除的特征
        valid_feat_names = [
            feat_name for feat_name in self.cfg.features.feat_names 
            if not self._should_exclude_feature(feat_name)
        ]
        names_set = set(valid_feat_names)

        # 处理变长稀疏特征
        for feat_name in self.cfg.features.varlen_sparse_feat_names:
            if self._should_exclude_feature(feat_name):
                logger.info(f"树模型跳过变长特征: {feat_name}")
                continue
                
            if feat_name in x.columns:
                x_explode = x[feat_name].apply(pd.Series)
                out_names = [feat_name + f"_{i}" for i in range(x_explode.shape[1])][:max_col_num]
                in_columns = [i for i in range(x_explode.shape[1])][:max_col_num]
                x[out_names] = pd.DataFrame(x_explode[in_columns], index=x.index)
                names_set.discard(feat_name)  # 移除原始特征名
                names_set = names_set.union(set(out_names))   # 添加展开后的特征名
        
        # 返回有效的特征列
        available_columns = [col for col in names_set if col in x.columns]
        logger.info(f"最终特征数量: {len(available_columns)}")
        
        return x[available_columns]

    def _prepare_input(self, trainset_cfg: TrainsetConfig):
        xs, ys = [], []
        logger.debug(f"trainset_cfg = {trainset_cfg}")
        for data_path in trainset_cfg.data_path:
            logger.info(f"loading data from {data_path}.")
            df = pd.read_csv(
                data_path,
                sep=trainset_cfg.sep,
                chunksize=trainset_cfg.chunksize,
                header=trainset_cfg.header,
                names=trainset_cfg.raw_columns,
            )
            logger.debug(f"train data: {df.head(10)}")
            x = df
            x = self._data_preprocess(x)
            feat = self._feat_selection(x, self.varlen_max)
            label = x[self.cfg.datasets.trainset.label_columns]
            xs.append(feat)
            ys.append(label)
        X = pd.concat(xs, axis=0)
        Y = pd.concat(ys, axis=0)
        logger.info(f"X.shape={X.shape}.Y.shape={Y.shape}.")
        return X, Y

    def train(self):
        X, Y = self._prepare_input(self.cfg.datasets.trainset)
        X_train, X_val, y_train, y_val = train_test_split(
            X.values, Y.values[:, 0], test_size=0.2, random_state=42
        )
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        # 记录训练开始时间
        logger.info("Starting model training...")
        
        # 添加训练集作为验证集的一部分，同时显示训练集和验证集的性能
        model = lgb.train(
            self.model_config.model_dump(),
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            feature_name=list(X.columns),
        )
        model.save_model("model.pth")
        
        # 使用模型对训练集和验证集进行预测，计算并打印AUC
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        train_auc = roc_auc_score(y_train, y_train_pred)
        val_auc = roc_auc_score(y_val, y_val_pred)
        
        logger.info(f"Final Training AUC: {train_auc:.4f}")
        logger.info(f"Final Validation AUC: {val_auc:.4f}")
        
        # 评估模型是否过拟合
        if train_auc - val_auc > 0.05:
            logger.warning(f"Possible overfitting detected: Train AUC - Val AUC = {train_auc - val_auc:.4f}")

        # 分析并输出特征重要性
        importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
        
        # 创建特征重要性数据框并排序
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        feature_importance_df = feature_importance_df.sort_values(
            by='Importance', ascending=False
        )
        
        # 保存特征重要性到文件
        importance_file = "feature_importance.csv"
        feature_importance_df.to_csv(importance_file, index=False)
        
        # 输出到日志
        logger.info("Top 20 important features:")
        for i, row in feature_importance_df.head(20).iterrows():
            logger.info(f"Feature {row['Feature']}: {row['Importance']}")
            
        # 检查是否有特征过于重要，可能表明数据泄露
        if feature_importance_df['Importance'].max() / feature_importance_df['Importance'].sum() > 0.5:
            logger.warning("Possible data leakage detected: A single feature contributes to more than 50% of the total importance")


if __name__ == "__main__":
    push_classifier = PushClassifier("config/config.yml", "config/model.yml")
    push_classifier.train()
