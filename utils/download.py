import datetime
import os
import shutil
from pathlib import Path
from typing import Dict

from hxhive.cli import hexecute
from loguru import logger


def remove_dir(path: Path, max_num: int):
    """保留指定目录下最新的max_num个文件，删除其余文件"""
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


def parse_sql(path: Path, now: str, last_1_day: str = None) -> str:
    """读取SQL文件并替换参数"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if last_1_day:
        content_formatted = content.format(now=now, last_1_day=last_1_day)
    else:
        content_formatted = content.format(now=now)
    return content_formatted


def get_multi_data_optimized(
    sql_path: Path,
    past_day: int,
    shift_day: int = 2,
    download: bool = True,
    max_file_num: int = 10,
    min_file_size: int = 1024,
) -> Dict[str, str]:
    """
    获取多天的数据（优化版）

    Parameters
    ----------
    sql_path : Path
        SQL文件路径
    past_day : int
        取前N天的数据
    shift_day : int, optional
        时间向前偏移, by default 2
    download : bool, optional
        是否从hive下载数据, by default True
    max_file_num : int, optional
        训练文件的最大数量, by default 10
    min_file_size : int, optional
        最小文件大小(KB), by default 1024

    Returns
    -------
    Dict[str, str]
        日期和对应SQL的字典
    """
    data_date = datetime.datetime.now() - datetime.timedelta(days=shift_day)
    logger.info(f"current date: {data_date.strftime('%Y%m%d')}")
    assert past_day > 0, "past_day must be greater than 0"

    sqls = dict()
    for i in range(past_day):
        # 计算当前日期
        p_date = (data_date - datetime.timedelta(days=past_day - 1 - i)).strftime(
            "%Y%m%d"
        )

        # 解析SQL
        sql = parse_sql(sql_path, p_date)
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

    # 清理旧文件
    train_folder = sql_path.parent / "train"
    os.makedirs(train_folder, exist_ok=True)
    logger.info(f"Current files in train folder: {os.listdir(train_folder)}")
    remove_dir(train_folder, max_num=max_file_num)
    return sqls


def download_from_temp_table(
    sql_path: Path,
    past_day: int = 7,
    shift_day: int = 0,
    download_dir: Path = None,
    min_file_size: int = 1024,
    max_file_num: int = 10,
):
    """
    从临时结果表中下载数据

    Parameters
    ----------
    sql_path : Path
        SQL文件路径
    past_day : int, optional
        需要下载前几天的数据, by default 7
    shift_day : int, optional
        时间偏移量（0表示今天）, by default 0
    download_dir : Path, optional
        下载目录, by default None (使用默认目录)
    min_file_size : int, optional
        最小文件大小(KB), by default 1024
    max_file_num : int, optional
        保留的最大文件数量, by default 10

    Returns
    -------
    bool
        是否成功下载了任何数据
    """
    # 设置默认下载目录
    if download_dir is None:
        download_dir = sql_path.parent / "train"

    # 确保下载目录存在
    os.makedirs(download_dir, exist_ok=True)

    success_count = 0

    # 下载指定天数的数据
    for i in range(past_day):
        # 计算要下载的日期
        data_date = datetime.datetime.now() - datetime.timedelta(days=i + shift_day)
        p_date = data_date.strftime("%Y%m%d")

        # 解析SQL
        sql = parse_sql(sql_path, p_date)
        logger.info(f"date={p_date}, sql=\n{sql}")

        # 设置下载文件路径
        file_path = download_dir / f"{p_date}.csv"

        try:
            # 执行SQL并下载数据
            logger.info(f"downloading data from hive to {file_path}")
            hexecute(sql, file_path)

            # 检查文件是否下载成功
            if (
                file_path.exists()
                and os.path.getsize(file_path) >= min_file_size * 1024
            ):
                logger.info(f"Successfully downloaded {file_path}")
                success_count += 1
            else:
                # 文件太小，可能是数据问题
                if file_path.exists():
                    os.remove(file_path)
                logger.warning(
                    f"File size smaller than {min_file_size}KB. Removed {file_path}"
                )
        except Exception as e:
            logger.error(f"Error downloading data for {p_date}: {e}")

    # 清理旧文件，保留最新的文件
    if max_file_num > 0:
        remove_dir(download_dir, max_num=max_file_num)

    logger.info(f"Successfully downloaded {success_count} out of {past_day} files")
    return success_count > 0


def download_user_data(data_dir: Path, sql_path: Path, sql_params: dict) -> bool:
    """
    下载用户数据并保存到指定目录。

    该函数首先读取SQL文件，将其内容格式化为特定的日期，然后执行SQL查询并将结果保存到指定目录。
    如果下载的数据文件大小大于4096字节，则认为下载成功；否则，删除文件并记录警告信息。

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


def get_multi_data(
    sql_path: Path,
    past_day: int,
    shift_day: int = 2,
    download: bool = True,
    max_file_num: int = 10,
    min_file_size: int = 1024,
) -> Dict[str, str]:
    """
    获取多天的数据 - 与train.py兼容的接口

    Parameters
    ----------
    sql_path : Path
        SQL文件路径
    past_day : int
        取前N天的数据
    shift_day : int, optional
        时间向前偏移, by default 2
    download : bool, optional
        是否从hive下载数据, by default True
    max_file_num : int, optional
        训练文件的最大数量, by default 10
    min_file_size : int, optional
        最小文件大小(KB), by default 1024

    Returns
    -------
    Dict[str, str]
        日期和对应SQL的字典
    """
    # 调用download_from_temp_table函数来实现功能
    download_from_temp_table(
        sql_path=sql_path,
        past_day=past_day,
        shift_day=shift_day,
        min_file_size=min_file_size,
        max_file_num=max_file_num,
    )

    # 为了兼容旧版接口，返回一个空字典
    return {}


if __name__ == "__main__":
    # 示例用法
    current_path = Path.cwd()
    # 使用train.sql文件，该文件的查询表名中已包含日期参数
    # 表名格式为: db_test.tmp_final_result_YYYYMMDD
    sql_path = current_path / "data" / "train.sql"
    download_from_temp_table(sql_path, past_day=7, shift_day=1, max_file_num=7)
