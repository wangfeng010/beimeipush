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
        model = lgb.train(
            self.model_config,
            train_data,
            valid_sets=[val_data],
            feature_name=list(X.columns),
        )
        model.save_model("model.pth")


if __name__ == "__main__":
    push_classifier = PushClassifier("config/config.yml", "config/model.yml")
    push_classifier.train()
