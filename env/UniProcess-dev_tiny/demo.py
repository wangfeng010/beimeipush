import csv
import time
from functools import partial

import yaml

from uniprocess.config import Config
from uniprocess.operations import OP_HUB


def run_one_operation(x, op):
    col_in = op.col_in
    col_out = op.col_out
    func_name = op.func_name
    parameters = op.func_parameters if op.func_parameters else dict()
    partial_func = partial(OP_HUB[func_name], **parameters)

    if isinstance(col_in, list):
        x[col_out] = partial_func(*[x[c] for c in col_in])
    else:
        x[col_out] = partial_func(x[col_in])
    return x


def get_one_feat(x, pipe):
    for op in pipe.operations:
        x = run_one_operation(x, op)
    return x


def test_csv_dataloader(cfg: Config, max_row: int):
    columns = cfg.datasets.trainset.raw_columns
    with open(cfg.datasets.trainset.data_path[0], "r") as file:
        csv_reader = csv.reader(file, delimiter=cfg.datasets.trainset.sep)
        for i, row in enumerate(csv_reader):
            if i == 0 and cfg.datasets.trainset.header:
                continue
            x = dict(zip(columns, row))
            pipelines = cfg.process.pipelines + cfg.interactions.pipelines

            for pipe in pipelines:
                x = get_one_feat(x, pipe)
            if i > max_row:
                return


def run_one_op_pd(x, op):
    col_in = op.col_in
    col_out = op.col_out
    func_name = op.func_name
    parameters = op.func_parameters if op.func_parameters else dict()
    partial_func = partial(OP_HUB[func_name], **parameters)

    if isinstance(col_in, list):
        x[col_out] = x[col_in].apply(lambda row: partial_func(*row), axis=1)
    else:
        x[col_out] = x[col_in].apply(partial_func)
    return x


def test_pandas_dataloader(cfg: Config, max_row: int):
    import pandas as pd

    df = pd.read_csv(cfg.datasets.trainset.data_path[0], sep="\t", chunksize=max_row)
    x = df.get_chunk()

    pipelines = cfg.process.pipelines + cfg.interactions.pipelines

    for pipe in pipelines:
        for op in pipe.operations:
            x = run_one_op_pd(x, op)


if __name__ == "__main__":
    with open("templates/demo_config.yml", encoding="utf-8", mode="r") as f:
        raw_config = yaml.safe_load(f)
    cfg = Config(**raw_config)
    MAX_ROW = 4096
    s = time.time()
    test_csv_dataloader(cfg, MAX_ROW)
    print(f"Process {MAX_ROW} rows. CSV loader cost:{time.time()-s}")
    s = time.time()
    test_pandas_dataloader(cfg, MAX_ROW)
    print(f"Process {MAX_ROW} rows. Pandas loader cost:{time.time()-s}")
