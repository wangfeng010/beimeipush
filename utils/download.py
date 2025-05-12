import datetime
import os
import shutil
from pathlib import Path
from typing import Dict

from hxhive.cli import hexecute
from loguru import logger


def remove_dir(path: Path, max_num: int):
    assert max_num > 0, "max_num must be greater than 0"
    # 当目录下文件数量超过max_num保留最新的max_num个文件
    file_num = len(list(path.iterdir()))
    logger.info(f"current path contains {file_num} files; max num: {max_num}")
    if file_num > max_num:
        for f in sorted(path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[
            max_num:  # 删除最旧的文件
        ]:
            if f.is_file():
                os.remove(f)
            else:
                shutil.rmtree(f)
            logger.warning(f"remove file: {f}")


def parse_sql(path: Path, now: str, last_1_day: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    content_formatted = content.format(now=now, last_1_day=last_1_day)
    return content_formatted


def get_multi_data(
    sql_path: Path,
    past_day: int,
    shift_day: int = 2,
    download: bool = True,
    max_file_num: int = 10,
    min_file_size: int = 1024,
) -> Dict[str, str]:
    """_summary_

    Parameters
    ----------
    sql_path : Path
        sql文本路径
    past_day : int
        取前N天的数据
    shift_day : int, optional
        时间向前偏移, by default 2
    download : bool, optional
        是否从hive下载数据, by default True
    max_file_num : int, optional
        训练文件的最大数量, by default 10

    Returns
    -------
    Dict[str, str]
        _description_
    """
    data_date = datetime.datetime.now() - datetime.timedelta(days=shift_day)
    p_date = data_date.strftime("%Y%m%d")
    logger.info(f"current date: {p_date}")
    last_1_day = (data_date - datetime.timedelta(days=1)).strftime("%Y%m%d")
    assert past_day > 1, "past_day must larger than 1"

    sqls = dict()
    for i in range(past_day + 1):
        p_date = (
            data_date - datetime.timedelta(days=past_day) + datetime.timedelta(days=i)
        ).strftime("%Y%m%d")
        last_1_day = (data_date - datetime.timedelta(days=i + 1)).strftime("%Y%m%d")
        sql = parse_sql(sql_path, p_date, last_1_day)
        sqls[p_date] = sql
        logger.info(f"date={p_date}, sql=\n{sql}")

        if download:
            download_dir = sql_path.parent / "train" / f"{p_date}.csv"
            logger.info(f"downloading data from hive to {download_dir}")
            hexecute(sql, download_dir)
            if (
                download_dir.exists()
                and os.path.getsize(download_dir) < min_file_size * 1024
            ):
                os.remove(download_dir)
                logger.warning(
                    f"file size smaller than {min_file_size * 1024}. remove {download_dir}"
                )

    train_folder = sql_path.parent / "train"
    logger.info(os.listdir(train_folder))
    remove_dir(sql_path.parent / "train", max_num=max_file_num)
    return sqls


def download_user_data(data_dir: Path, sql_path: Path, sql_params: dict) -> bool:
    """
    下载用户数据并保存到指定目录。

    该函数首先读取SQL文件，将其内容格式化为特定的日期，然后执行SQL查询并将结果保存到指定目录。
    如果下载的数据文件大小大于4096字节，则认为下载成功；否则，删除文件并记录警告信息。
    如果目录不存在，记录错误信息。

    :param data_dir: 保存下载数据的目录路径。
    :param sql_path: 包含SQL查询的文件路径。
    :param sql_params: 用于格式化SQL查询的日期字符串。
    :return: 如果下载成功返回True，否则返回False。
    """
    # 读取SQL文件内容
    with open(sql_path, "r", encoding="utf-8") as f:
        raw_sql = f.read()

    # 格式化SQL查询字符串
    sql = raw_sql.format(**sql_params)
    logger.info(f"sql:\n{sql}")
    try:
        # 执行SQL查询并将结果保存到指定目录
        hexecute(sql, data_dir)
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        return False

    # 检查数据目录是否存在
    if data_dir.exists():
        # 检查文件大小是否大于4096字节
        if os.path.getsize(data_dir) > 4096:
            logger.info(f"dowloading {data_dir} success.")
            return True
        else:
            # 如果文件大小过小，删除文件并记录警告信息
            os.remove(data_dir)
            logger.warning(f"{data_dir} file size too small. delete.")
    else:
        # 如果目录不存在，记录错误信息
        logger.error(f"dowloading {data_dir} fail.")

    return False


if __name__ == "__main__":
    current_path = Path.cwd()
    train_path = current_path / "data" / "train.sql"
    content_formatted = get_multi_data(train_path, 5, max_file_num=5)
