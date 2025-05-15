import lightgbm as lgb
import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split
from uniprocess.config import Config

from utils.preprocess import run_one_op_pd


class PushClassifier:
    def __init__(
        self,
        config_dir: str,
        model_config_dir: str,
        mode: str = "train",
        varlen_max: int = 5,
    ) -> None:
        self.cfg = self._get_config(config_dir)
        self.varlen_max = varlen_max

        with open(model_config_dir, encoding="utf-8", mode="r") as f:
            raw_config = yaml.safe_load(f)
        self.model_config = raw_config

    def _get_config(self, config_dir: str) -> Config:
        with open(config_dir, encoding="utf-8", mode="r") as f:
            raw_config = yaml.safe_load(f)
        cfg = Config(**raw_config)
        logger.info(f"base config: \n{cfg}")
        return cfg

    def _data_preprocess(self, x: pd.DataFrame) -> pd.DataFrame:
        # 特征
        pipelines = self.cfg.process.pipelines + self.cfg.interactions.pipelines
        for pipe in pipelines:
            for op in pipe.operations:
                x = run_one_op_pd(x, op)
        # label
        for pipe in self.cfg.label_process.pipelines:
            for op in pipe.operations:
                x = run_one_op_pd(x, op)
        out_columns = self.cfg.feat_names + self.cfg.datasets.trainset.label_columns
        return x[out_columns]

    def _feat_selection(self, x: pd.DataFrame, max_col_num: int) -> pd.DataFrame:
        names_set = set(self.cfg.feat_names)

        for feat_name in self.cfg.model.varlen_sparse_feat_names:
            x_explode = x[feat_name].apply(pd.Series)
            out_names = [feat_name + f"_{i}" for i in range(x_explode.columns.stop)][
                :max_col_num
            ]
            in_columns = [i for i in range(x_explode.columns.stop)][:max_col_num]
            x[out_names] = pd.DataFrame(x_explode[in_columns], index=x.index)
            names_set.remove(feat_name)
            names_set = names_set.union(set(out_names))
        return x[list(names_set)]

    def _prepare_input(self, trainset_cfg):
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
            
            # 预处理时间格式: 统一处理带毫秒和不带毫秒的时间戳
            def normalize_timestamp(time_str):
                if isinstance(time_str, str):
                    if '.' in time_str:  # 包含毫秒
                        # 移除毫秒部分
                        return time_str.split('.')[0]
                return time_str
                
            # 应用时间格式标准化
            x = df
            if 'create_time' in x.columns:
                x['create_time'] = x['create_time'].apply(normalize_timestamp)
                logger.info(f"标准化时间格式完成，示例: {x['create_time'].head(3).tolist()}")
                
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

        # 配置日志文件
        log_file = "lgb_train.log"

        # 配置回调函数列表
        callbacks = [
            lgb.callback.log_evaluation(period=10),  # 每10次迭代记录一次
        ]

        # 打开文件用于写入日志
        with open(log_file, "w") as f:
            # 创建回调函数，将输出写入文件
            def callback_log(env):
                iteration = env.iteration
                val_auc = env.evaluation_result_list[0][2]
                f.write(f"Iteration: {iteration}, val AUC: {val_auc}\n")
                f.flush()  # 确保立即写入

            callbacks.append(callback_log)

            model = lgb.train(
                self.model_config,
                train_data,
                valid_sets=[val_data],
                feature_name=list(X.columns),
                callbacks=callbacks,
            )

        model.save_model("model.pth")

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

        return log_file  # 返回日志文件路径


if __name__ == "__main__":
    push_classifier = PushClassifier("config/config.yml", "config/model.yml")
    push_classifier.train()
