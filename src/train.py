import zipfile
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.model import PushClassifier
from src.config.base_config import AppConfig
from src.config.yaml_config import load_config_from_yaml


def upload():
    cur_path = Path.cwd()
    model_file = cur_path / "model.pth"
    config_file = cur_path / "config" / "config.yml"

    config: AppConfig = load_config_from_yaml(config_file, AppConfig)
    HDFS_DIR = config.hdfs.base_model_hdfs_dir + config.hdfs.model_version
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    archive_name = f"model_and_config_{timestamp}.zip"
    # 1. 打包模型文件和配置文件
    logger.info(f"start packaging {model_file}, {config_file},into {archive_name}.")
    with zipfile.ZipFile(archive_name, "w") as zipf:
        zipf.write(model_file, arcname=model_file.name)
        zipf.write(config_file, arcname=config_file.relative_to(cur_path))
    logger.info("packaging finished.")

    # 2. 上传打包文件
    logger.info(f"start uploading {archive_name} to {HDFS_DIR}.")
    from src.data.hdfs import upload_hdfs

    upload_hdfs(HDFS_DIR, str(archive_name), remote_url=config.hdfs.api_url)
    logger.info("upload finished.")

    # 3. 上传完成后下载，测试是否成功
    download_path = "tmp_download.zip"
    from src.data.hdfs import download_hdfs

    download_hdfs(
        f"{HDFS_DIR}/{archive_name}",
        Path(download_path),
        remote_url=config.hdfs.api_url,
    )
    if Path(download_path).exists():
        logger.info("upload and download test success.")
        Path(download_path).unlink()
    else:
        logger.error("download test fail.")


def main():
    cur_path = Path.cwd()
    # 2. 模型训练
    model = PushClassifier(
        config_dir=cur_path / "config" / "config.yml",
        mode="trian",
        varlen_max=5,
    )
    model.train()

    upload()


if __name__ == "__main__":
    main()
