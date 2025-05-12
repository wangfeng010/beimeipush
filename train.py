from pathlib import Path

from loguru import logger

from model import PushClassifier


def data_download(cur_path: Path, data_num: int):
    logger.info("downloading data from hive.")
    from utils.download import get_multi_data

    sql_path = cur_path / "data" / "train.sql"
    get_multi_data(sql_path, past_day=data_num)


def main():
    HDFS_DIR = "dongshaojie/ainvest_push/tree_model"
    # 1.下载数据
    cur_path = Path.cwd()
    # 2. 模型训练
    model = PushClassifier(
        config_dir=cur_path / "config" / "config.yml",
        model_config_dir=cur_path / "config" / "model.yml",
        mode="trian",
        varlen_max=5,
    )
    model.train()
    # 3. 上传模型
    logger.info("start uploading.")
    from utils.hdfs import upload_hdfs

    upload_hdfs(
        HDFS_DIR,
        "model.pth",
    )

    model_path = "tmp.pth"
    from utils.hdfs import download_hdfs

    download_hdfs(HDFS_DIR, model_path)
    if Path(model_path).exists():
        logger.info("upload success.")
    else:
        logger.error("download fail.")


if __name__ == "__main__":
    main()
