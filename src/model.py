import lightgbm as lgb
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config.base_config import AppConfig, TrainsetConfig
from src.config.yaml_config import load_config_from_yaml
from src.preprocess.preprocess import run_one_op_pd


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

    def _get_config(self, config_dir: str) -> AppConfig:
        config = load_config_from_yaml(config_dir, AppConfig)

        logger.info(f"base config: \n{config}")
        return config

    def _data_preprocess(self, x: pd.DataFrame) -> pd.DataFrame:
        # 特征
        pipelines = (
            self.cfg.features.process.pipelines
            + self.cfg.features.interactions.pipelines
        )
        for pipe in pipelines:
            for op in pipe.operations:
                try:
                    x = run_one_op_pd(x, op)
                except Exception as e:
                    logger.debug(
                        f"input:\n{x[op.col_in]}. op:{op.col_in}. got error{e}"
                    )
                    raise e
        # label
        for pipe in self.cfg.features.label_process.pipelines:
            for op in pipe.operations:
                x = run_one_op_pd(x, op)
        out_columns = (
            self.cfg.features.feat_names + self.cfg.datasets.trainset.label_columns
        )
        return x[out_columns]

    def _feat_selection(self, x: pd.DataFrame, max_col_num: int) -> pd.DataFrame:
        names_set = set(self.cfg.features.feat_names)

        for feat_name in self.cfg.features.varlen_sparse_feat_names:
            x_explode = x[feat_name].apply(pd.Series)
            out_names = [feat_name + f"_{i}" for i in range(x_explode.columns.stop)][
                :max_col_num
            ]
            in_columns = [i for i in range(x_explode.columns.stop)][:max_col_num]
            x[out_names] = pd.DataFrame(x_explode[in_columns], index=x.index)
            names_set.remove(feat_name)
            names_set = names_set.union(set(out_names))
        return x[list(names_set)]

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
