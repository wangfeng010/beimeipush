from pathlib import Path

from loguru import logger

from model import PushClassifier

HDFS_DIR = "dongshaojie/ainvest_push/tree_model"


def upload(files_to_upload=None):
    """上传模型和日志到HDFS并验证

    Args:
        files_to_upload: 需要上传的额外文件列表
    """
    logger.info("start uploading.")
    from utils.hdfs import upload_hdfs

    # 上传模型文件
    upload_hdfs(
        HDFS_DIR,
        "model.pth",
    )

    # 上传额外的日志文件
    if files_to_upload:
        for file in files_to_upload:
            if Path(file).exists():
                logger.info(f"uploading log file: {file}")
                upload_hdfs(HDFS_DIR, file)
                logger.info(f"log file {file} uploaded")

    # 验证上传
    model_path = "tmp.pth"
    from utils.hdfs import download_hdfs

    download_hdfs(HDFS_DIR, model_path)
    if Path(model_path).exists():
        logger.info("upload success.")
    else:
        logger.error("download fail.")


def main():
    # 1.下载数据
    cur_path = Path.cwd()

    # 2. 模型训练
    model = PushClassifier(
        config_dir=cur_path / "config" / "config.yml",
        model_config_dir=cur_path / "config" / "model.yml",
        mode="train",
        varlen_max=5,
    )

    # 训练模型并获取日志文件路径
    log_file = model.train()
    logger.info(f"Model training completed, log file: {log_file}")

    # 3. 上传模型和日志
    # upload([log_file])


if __name__ == "__main__":
    main()
