from pathlib import Path

import requests

URL = "http://ainvest-recsys-platform-server-0-prod.default:9595/recsys/api/bigFileUploader/"
DOWNLOAD_URL = URL + "download"


def upload_hdfs(hdfs_path: str, local_file: str) -> requests.Response:
    """上传至HDFS

    Parameters
    ----------
    filePath : str
        hdfs目录
    fileName : Path
        本地目录

    Returns
    -------
    _type_
        _description_
    """
    url = URL + "upload"
    d = {"dirName": hdfs_path}
    files = {"file": (local_file, open(local_file, "rb"), "application/octet-stream")}
    return requests.post(url, data=d, files=files)


def download_hdfs(
    hdfs_path: str, local_path: Path, remote_url: str = DOWNLOAD_URL
) -> bool:
    """从hdfs下载文件

    Parameters
    ----------
    hdfs_path : str
        hdfs目录，目录下必须只有一个文件
    local_path : Path
        保存的本地文件名

    Returns
    -------
    bool
        _description_
    """
    params = {"dirName": hdfs_path}
    response = requests.get(remote_url, params=params)
    if response.content is None:
        return False
    with open(local_path, "wb") as f:
        f.write(response.content)
    return True
