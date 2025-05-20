from pathlib import Path

import requests


def upload_hdfs(hdfs_path: str, local_file: str, remote_url: str) -> requests.Response:
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
    url = remote_url + "upload"
    d = {"dirName": hdfs_path}
    files = {"file": (local_file, open(local_file, "rb"), "application/octet-stream")}
    return requests.post(url, data=d, files=files)


def download_hdfs(hdfs_path: str, local_path: Path, remote_url: str) -> bool:
    params = {"dirName": hdfs_path}
    url = remote_url + "download"
    response = requests.get(url, params=params)
    if response.content is None:
        return False
    with open(local_path, "wb") as f:
        f.write(response.content)
    return True
