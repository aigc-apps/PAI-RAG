import logging
import os
import json

import oss2

logger = logging.getLogger(__name__)


class OssCache:
    def __init__(self, oss_config: json) -> None:
        access_key_id = oss_config["access_key_id"]
        access_key_secret = oss_config["access_key_secret"]
        bucket_name = oss_config["bucket_name"]
        endpoint = oss_config["endpoint"]
        """
        确认上面的参数都填写正确了,如果任何一个参数包含 '<'，意味着这个参数可能没有被正确设置，而是保留了一个占位符或默认值（
        这通常在配置文件中使用 <your-access-key-id> 这样的占位符来表示未设置的参数）。
        """
        for param in (access_key_id, access_key_secret, bucket_name, endpoint):
            assert "<" not in param, "请设置参数：" + param
        self.bucket = oss2.Bucket(
            oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name
        )

    def get_object(self, key: str) -> bytes:
        if self.bucket.object_exists(key):
            logger.info("file exists")
            return self.bucket.get_object(key)
        else:
            logger.info("file does not exist")
            return None

    def put_object(self, key: str, data: bytes) -> None:
        self.bucket.put_object(key, data)

    def upload_files(self, local_folder: str, oss_folder: str) -> None:
        logger.info(f"Start to upload from {local_folder} to {oss_folder}")
        for root, dirs, files in os.walk(local_folder):
            for file_name in files:
                local_file_path = os.path.join(root, file_name)
                oss_file_path = os.path.join(
                    oss_folder, os.path.relpath(local_file_path, start=local_folder)
                )
                # 上传文件
                self.bucket.put_object_from_file(oss_file_path, local_file_path)
                logger.info(f"Uploaded {local_file_path} to {oss_file_path}")
        logger.info(f"Finished uploading files from {local_folder} to {oss_folder}")

    def download_files(self, oss_folder: str, local_folder: str) -> None:
        logger.info(f"Start to download from {oss_folder} to {local_folder}")
        # List all files on OSS folder
        for obj in oss2.ObjectIterator(self.bucket, prefix=oss_folder):
            if obj.is_prefix():  # 一个目录/文件夹
                continue  # 可以选择递归下载，迭代处理子文件夹
            else:  # 一个文件
                local_file_path = os.path.join(
                    local_folder, os.path.relpath(obj.key, oss_folder)
                )
                # 确保文件的目录存在
                local_file_dir = os.path.dirname(local_file_path)
                if not os.path.exists(local_file_dir):
                    os.makedirs(local_file_dir)
                # 开始下载
                self.bucket.get_object_to_file(obj.key, local_file_path)
                logger.info(f"Downloaded {obj.key} to {local_file_path}")
        logger.info(f"Finished downloading files from {oss_folder} to {local_folder}")
