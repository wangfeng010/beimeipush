import os
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import lightgbm
import pandas as pd
import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import BackgroundTasks, FastAPI
from loguru import logger
from uniprocess.config import Config

from model import run_one_op_pd
from utils.download import download_user_data
from utils.dtypes import InferConfig, PushItems
from utils.hdfs import download_hdfs
from utils.preprocess import data_preprocess

resource = {}


def daily_task():
    """
    每日任务：更新用户数据。

    该方法执行以下步骤：
    1. 从资源中获取配置信息和推理配置。
    2. 计算前两天的日期和当前日期。
    3. 构建当前日期的用户数据文件路径。
    4. 获取数据目录下所有 CSV 文件，并按时间倒序排序。
    5. 如果没有数据文件或当天文件不存在，尝试下载数据，最多尝试 `download_max_try` 次。
    6. 重新获取数据目录下所有 CSV 文件，并按时间倒序排序。
    7. 保留最近 `user_data_save_days` 天的数据文件，删除其他文件。
    8. 如果当天未下载成功但有文件存在，使用最新的文件。
    9. 如果当天未下载成功且没有文件存在，记录错误并返回。
    10. 读取 CSV 文件并设置分隔符、表头和列名。
    11. 设置数据索引。
    12. 对数据进行预处理。
    13. 重置索引。
    14. 记录用户数据的形状。
    15. 更新资源中的用户数据。
    """
    config = resource["config"]
    infer_config: InferConfig = resource["infer_config"]

    # 计算前一天的日期和当前日期
    last_1_day = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d")
    cur_date = datetime.now().strftime("%Y%m%d")

    # 构建数据文件路径
    data_dir = Path(infer_config.user_data_dir) / f"{cur_date}.csv"

    # 获取数据目录下的所有CSV文件，并按时间倒序排序
    file_list = sorted(list(data_dir.parent.glob("*.csv")), reverse=True)

    # 初始化下载标志
    flag = False

    # 如果没有数据文件或当天文件不存在，尝试下载数据
    if not file_list or (data_dir not in file_list):
        for i in range(infer_config.download_max_try):
            flag = download_user_data(
                data_dir=data_dir,
                sql_path=infer_config.user_data_sql_dir,
                sql_params={"now": last_1_day},
            )

            if flag:
                logger.info(f"download {data_dir} success at {i}-th try.")
                break

    # 重新获取数据目录下的所有CSV文件，并按时间倒序排序
    file_list = sorted(list(data_dir.parent.glob("*.csv")), reverse=True)

    # 保留近N天的数据文件，其他的删除
    if len(file_list) > infer_config.user_data_save_days:
        for file in file_list[infer_config.user_data_save_days :]:
            os.remove(file)
            logger.warning(f"file number exceeded. remove {file}")

    # 如果当天未下载成功，但有文件存在，使用最新的文件
    if not flag and len(file_list) > 0:
        data_dir = file_list[0]

    # 如果当天未下载成功且没有文件存在，记录错误并返回
    if not flag and len(file_list) == 0:
        logger.error("no user data.")
        return

    # 记录读取数据的日志
    logger.info(f"reading data: {data_dir}")

    # 读取CSV文件并设置分隔符、表头和列名
    df = pd.read_csv(
        data_dir,
        sep=infer_config.sep,
        header=infer_config.header,
        names=infer_config.data_columns,
    )

    # 设置数据索引
    df = df.set_index(infer_config.data_index)
    logger.debug(f"loaded user df, shape={df.shape}: {df.head()}")
    # 数据预处理
    user_df = data_preprocess(df, config, infer_config)

    # 重置索引
    user_df = user_df.reset_index()

    # 记录用户数据形状的日志
    logger.info(f"user data shape: {user_df.shape}")

    # 更新资源中的用户数据
    resource.update({"user_data": user_df})


def hour_task():
    """
    每小时任务：下载新用户数据并进行预处理。

    该方法执行以下步骤：
    1. 记录任务开始日志。
    2. 从资源中获取配置信息和推理配置。
    3. 计算当前日期、前一天日期和前两天日期。
    4. 构建新用户数据文件路径。
    5. 从 HDFS 下载新用户数据。
    6. 如果下载失败或数据文件不存在，记录错误并返回。
    7. 读取 CSV 文件并设置分隔符、表头和列名。
    8. 设置数据索引。
    9. 对数据进行预处理。
    10. 记录新用户数据的形状。
    11. 获取当前用户数据并合并新用户数据。
    12. 去重并更新资源中的用户数据。
    """
    logger.info("start hour task: download new user data.")
    config = resource["config"]
    infer_config: InferConfig = resource["infer_config"]

    # 计算前一天的日期和当前日期
    today = datetime.now().strftime("%Y%m%d")
    last_1_day = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    last_2_day = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d")

    # 构建数据文件路径
    data_dir = Path(infer_config.new_user_data_dir) / "new_user.csv"

    # 下载新用户数据
    flag = download_user_data(
        data_dir=data_dir,
        sql_path=infer_config.new_user_data_sql_dir,
        sql_params={"today": today, "last_1_day": last_1_day, "now": last_2_day},
    )

    # 如果当天未下载成功且没有文件存在，记录错误并返回
    if not flag:
        logger.error("no new user data.")
        return

    # 记录读取数据的日志
    logger.info(f"reading data: {data_dir}")

    # 读取CSV文件并设置分隔符、表头和列名
    df = pd.read_csv(
        data_dir,
        sep=infer_config.sep,
        header=infer_config.header,
        names=infer_config.data_columns,
    )

    # 设置数据索引
    df = df.set_index(infer_config.data_index)

    # 数据预处理
    new_user_df = data_preprocess(df, config, infer_config).reset_index()

    # 记录用户数据形状的日志
    logger.info(f"new user data shape: {new_user_df.shape}")
    cur_df = resource.get("user_data", pd.DataFrame())
    logger.info(f"user data shape: {cur_df.shape}")
    cur_df = pd.concat([cur_df, new_user_df], axis=0, ignore_index=True)
    cur_df = cur_df.groupby("user_id").agg(max).reset_index()
    logger.info(f"merge user and current new user, data shape={cur_df.shape}")
    resource.update({"user_data": cur_df})


def load_model(infer_config: InferConfig) -> bool:
    """
    加载模型。

    该方法从配置中获取模型的 HDFS 路径和本地路径，尝试从 HDFS 下载模型到本地。
    - 如果下载成功，加载 LightGBM 模型，并将其更新到资源缓存中。
    - 如果下载或加载失败，返回 `False`；否则，返回 `True`。

    @param infer_config: 推理配置对象，包含模型的 HDFS 和本地路径。
    @return: 是否成功加载模型的布尔值。
    """
    model_dir = Path(infer_config.ml_model_dir)
    try:
        download_hdfs(
            infer_config.ml_model_hdfs_dir,
            model_dir,
        )
    except Exception as e:
        logger.error(e)
    if model_dir.exists():
        logger.info(f"dowloading model to {model_dir} success.")
    else:
        logger.error(f"dowloading model to {model_dir} fail.")
        return False
    model = lightgbm.Booster(model_file=infer_config.ml_model_dir)

    resource.update({"model": model, "feature_name": model.feature_name()})
    logger.info("update model to cache.")
    return True


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    current_path = Path().cwd()
    LOG_DIR = current_path / "log" / "item_infer.log"
    INFER_CONFIG_DIR = current_path / "config" / "infer.yml"
    CONFIG_DIR = current_path / "config" / "config.yml"

    logger.add(LOG_DIR, rotation="10 MB", level="INFO")
    logger.info("start.")

    logger.info("1.loading configuration.")
    with open(INFER_CONFIG_DIR, "r") as f:
        infer_config = yaml.safe_load(f)
    infer_config = InferConfig.model_validate(infer_config)
    resource.update({"infer_config": infer_config})
    with open(CONFIG_DIR, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    base_config = Config(**config_dict)
    resource.update({"config": base_config})
    logger.info(f"base config:\n{base_config}.\n infer config:\n{infer_config}.")

    logger.info("2. init daily task.")
    scheduler = BackgroundScheduler()
    trigger = CronTrigger(
        hour=infer_config.dowload_start_hour, minute=infer_config.dowload_start_minute
    )
    hour_trigger = CronTrigger(minute=infer_config.hour_task_minute)
    scheduler.add_job(daily_task, trigger=trigger)
    scheduler.add_job(hour_task, trigger=hour_trigger)
    scheduler.start()
    daily_task()
    hour_task()

    logger.info("3.loading model.")
    try:
        load_model(infer_config)
    except Exception as e:
        logger.error(e)

    yield
    resource.clear()
    scheduler.shutdown()
    logger.info("clean up lifespan")


app = FastAPI(lifespan=app_lifespan)


def data_to_feat(
    df_join: pd.DataFrame, base_config: Config, infer_config: InferConfig
) -> pd.DataFrame:
    """
    将数据转换为特征 DataFrame。

    该方法根据配置文件对输入的 DataFrame 进行特征工程处理。具体步骤包括：
    1. 根据 `infer_config` 中的 `varlen_max_col_num` 设置最大列数。
    2. 遍历 `base_config` 中定义的每条流水线（pipeline）。
    3. 对每条流水线中的每个操作（operation）进行处理，更新 DataFrame。
    4. 如果流水线的特征类型为 `varlen_sparse`，将特征列拆分为多个子列，并限制子列数量不超过 `max_col_num`。
    5. 最后，根据 `resource` 中定义的特征名称列表 `feature_name` 提取所需的特征列。

    @param df_join: 输入的 DataFrame，包含原始数据。
    @param base_config: 基础配置对象，定义了特征处理的流水线和操作。
    @param infer_config: 推理配置对象，包含最大列数等参数。
    @return: 处理后的特征 DataFrame。
    """
    max_col_num = infer_config.varlen_max_col_num
    for pipe in base_config.interactions.pipelines:
        for op in pipe.operations:
            df_join = run_one_op_pd(df_join, op)
        if pipe.feat_type == "varlen_sparse":
            x_explode = df_join[pipe.feat_name].apply(pd.Series)
            out_names = [
                pipe.feat_name + f"_{i}" for i in range(x_explode.columns.stop)
            ][:max_col_num]
            in_columns = [i for i in range(x_explode.columns.stop)][:max_col_num]
            df_join[out_names] = pd.DataFrame(
                x_explode[in_columns], index=df_join.index
            )
    feat_names = resource.get("feature_name")
    return df_join[feat_names]


def predict_and_append(df, model, df_join, results):
    """
    预测并添加结果到结果列表中。

    @param df: 需要预测的 DataFrame。
    @param model: 模型对象。
    @param df_join: 合并的 DataFrame，包含用户ID和项目ID。
    @param results: 结果列表，用于存储预测结果。
    """
    y_pred = model.predict(df.values)
    tmp = df_join[["user_id", "item_id"]]
    tmp["y"] = y_pred
    results.append(tmp)


def predict_items(item_df: pd.DataFrame, filter: bool = False) -> pd.DataFrame:
    """
    预测项目推荐结果。

    该方法执行以下步骤：
    1. 从资源中获取配置信息和推理配置。
    2. 记录项目数据的形状。
    3. 对项目数据进行预处理。
    4. 获取用户数据。
    5. 计算用户数据的批次索引。
    6. 初始化结果列表和交集对列表。
    7. 按批次处理用户数据：
       - 将当前批次的用户数据与项目数据进行笛卡尔积合并。
       - 将合并后的数据转换为模型特征。
       - 使用训练好的模型进行预测。
       - 根据是否过滤交集，分别处理有交集和无交集的数据。
    8. 合并所有批次的结果。
    9. 如果开启过滤，返回包含交集对的结果；否则，仅返回预测结果。

    @param item_df: 项目数据 DataFrame。
    @param filter: 是否进行交集过滤，默认为 False。
    @return: 预测结果 DataFrame。
    """
    # 从资源中获取配置和推理配置
    config = resource["config"]
    infer_config: InferConfig = resource["infer_config"]

    # 记录项目数据的形状
    logger.info(f"item shape: {item_df.shape}")

    # 对项目数据进行预处理
    item_df = data_preprocess(item_df, config, infer_config)
    logger.info("item data process end.")

    # 获取用户数据
    user_df: pd.DataFrame = resource.get("user_data")
    logger.debug(f"user info: shape={user_df.shape}, head={user_df.head()}")
    # 获取用户数量
    user_num = user_df.shape[0]

    # 计算批次索引
    indices = list(range(0, user_num, infer_config.max_user_num_per_iter))
    left = indices
    right = indices[1:] + [user_num]

    # 初始化结果列表
    results = []
    indices_names = ["user_id", "item_id"]
    # 按批次处理用户数据
    for l, r in zip(left, right):
        logger.info(f"predicting user from {l} to {r}.")

        # 将当前批次的用户数据与项目数据进行笛卡尔积合并
        df_join = user_df[l:r].join(item_df, how="cross")
        logger.info(f"user join item, got data shape: {df_join.shape}")

        # 将合并后的数据转换为模型特征
        df_feat = data_to_feat(df_join, config, infer_config)

        # 使用训练好的模型进行预测
        model: lightgbm.Booster = resource.get("model")

        if filter:
            df_feat["user_id"] = df_join["user_id"]
            df_feat["item_id"] = df_join["item_id"]
            # 交集判断
            condition = (
                (df_feat["preder_bid_cross"] == 1)
                | (df_feat["watch_bid_cross"] == 1)
                | (df_feat["hold_bid_cross"] == 1)
            )
            df_without_intersection = df_feat[~condition]
            df_with_intersection = df_feat[condition]

            # 根据 df_with_intersection中的user_id，从df_without_intersection删除

            user_ids_in_cross = df_with_intersection["user_id"].unique()
            condition2 = df_without_intersection["user_id"].isin(user_ids_in_cross)
            df_without_intersection = df_without_intersection[~condition2]

            df_in = pd.concat([df_with_intersection, df_without_intersection], axis=0)
            df_idices = df_in[indices_names]
            logger.debug(f"model input data shape={df_in.shape}")
            y_pred = model.predict(df_in.drop(columns=indices_names).values)
            df_idices["y"] = y_pred
            results.append(df_idices)
        else:
            predict_and_append(df_feat, model, df_join, results)
    # 合并所有批次的结果
    results = pd.concat(results, axis=0, ignore_index=True)

    return results


def ml_predict(items: PushItems, filter: bool = False):
    """
    预测给定项目的推荐结果。

    该方法首先将输入的项目转换为DataFrame格式，然后进行预测，最后返回每个项目的推荐用户列表。

    @param items: 包含待预测项目的对象。
    @return: 推荐结果，字典形式，键为项目ID，值为推荐用户的列表。
    """
    logger.info("predict start.")
    res = []

    item_dict = defaultdict(list)
    for item in items.items:
        item_feat = item.model_dump()
        for k, v in item_feat.items():
            item_dict[k].append(v)
    item_df = pd.DataFrame(item_dict)
    logger.debug(f"item: {item_df.head()}")
    y = predict_items(item_df, filter)
    # 找到每个用户的最佳推荐项目
    res = y.loc[y.groupby("user_id")["y"].idxmax()]
    res = res.groupby("item_id")["user_id"].apply(list).to_dict()
    item_user_num = {k: len(v) for k, v in res.items()}

    logger.info(f"got {len(items.items)} item, return {len(res)} item")
    logger.info(f"user number of each item:\n {item_user_num}")
    logger.info("predict success.")

    return res


@app.post("/predict/")
async def predict(items: PushItems):
    """物品向量化接口

    Parameters
    ----------
    item : Item
        _description_

    Returns
    -------
    ItemVector
        _description_
    """
    return ml_predict(items)


@app.post("/predict_v2/")
async def predict_v2(items: PushItems):
    """物品向量化接口

    Parameters
    ----------
    item : Item
        _description_

    Returns
    -------
    ItemVector
        _description_
    """
    return ml_predict(items, filter=True)


async def predict_send(items: PushItems):
    """
    异步发送预测结果到指定的回调URL。

    该方法首先调用 `ml_predict` 函数获取预测结果，然后使用 aiohttp 库将结果以 JSON 格式 POST 到配置中指定的回调URL，并返回响应文本。

    @param items: 包含待预测项目的对象。
    @return: 回调URL的响应文本。
    """
    res = ml_predict(items)
    callback_url = resource.get("infer_config").push_server_url
    async with aiohttp.ClientSession() as session:
        async with session.post(callback_url, json=res) as response:
            return await response.text()


async def predict_send_v2(items: PushItems):
    """
    异步发送预测结果到指定的回调URL。

    该方法首先调用 `ml_predict` 函数获取预测结果，然后使用 aiohttp 库将结果以 JSON 格式 POST 到配置中指定的回调URL，并返回响应文本。

    @param items: 包含待预测项目的对象。
    @return: 回调URL的响应文本。
    """
    res = ml_predict(items, filter=True)
    callback_url = resource.get("infer_config").push_server_url_v3
    async with aiohttp.ClientSession() as session:
        async with session.post(callback_url, json=res) as response:
            return await response.text()


@app.post("/predict_async/")
async def predict_async(items: PushItems, background_tasks: BackgroundTasks):
    """
    异步处理项目预测请求。

    该方法接收一个包含待预测项目的对象，将其传递给 `background_tasks`，
    并调用 `predict_send` 方法在后台异步处理预测结果的发送。返回一个包含接收到的项目数量和状态信息的字典。

    @param items: 包含待预测项目的对象。
    @param background_tasks: FastAPI 提供的后台任务管理器，用于异步执行任务。
    @return: 包含接收到的项目数量和状态信息的字典。
    """
    background_tasks.add_task(predict_send, items)

    return {"message": f"{len(items.items)} received.", "state": "200"}


@app.post("/predict_async_v2/")
async def predict_async_v2(items: PushItems, background_tasks: BackgroundTasks):
    """
    异步处理项目预测请求。

    该方法接收一个包含待预测项目的对象，将其传递给 `background_tasks`，
    并调用 `predict_send` 方法在后台异步处理预测结果的发送。返回一个包含接收到的项目数量和状态信息的字典。

    @param items: 包含待预测项目的对象。
    @param background_tasks: FastAPI 提供的后台任务管理器，用于异步执行任务。
    @return: 包含接收到的项目数量和状态信息的字典。
    """
    background_tasks.add_task(predict_send_v2, items)

    return {"message": f"{len(items.items)} received.", "state": "200"}


@app.post("/update_model")
async def update_model():
    """
    更新模型。

    该方法从配置中获取推理配置，并调用 `load_model` 函数加载新的模型。
        - 如果加载成功，返回状态为 "success"；否则，返回状态为 "fail"。

    @return: 包含更新状态的字典。
    """
    infer_config: InferConfig = resource["infer_config"]
    flag = load_model(infer_config)
    if flag:
        return {"state": "success"}
    else:
        return {"state": "fail"}


@app.get("/readiness")
async def readiness():
    """
    检查服务的就绪状态。

    该方法用于检查服务是否已准备好处理请求。如果服务正常运行，返回状态码为0和状态信息为 "OK" 的字典。

    @return: 包含就绪状态的字典。
    """
    return {"code": 0, "status": "OK"}
