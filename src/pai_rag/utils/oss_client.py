import logging
import hashlib

import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

logger = logging.getLogger(__name__)


class OssClient:
    def __init__(self, bucket_name: str, endpoint: str):
        auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
        self.bucket_name = bucket_name
        self.endpoint = endpoint
        self.base_url = self._make_url()

        """
        确认上面的参数都填写正确了,如果任何一个参数包含 '<'，意味着这个参数可能没有被正确设置，而是保留了一个占位符或默认值（
        这通常在配置文件中使用 <your-access-key-id> 这样的占位符来表示未设置的参数）。
        """
        for param in (bucket_name, endpoint):
            assert "<" not in param, "请设置参数：" + param
        self.bucket = oss2.Bucket(auth, endpoint, bucket_name)

    def exists(self, key: str) -> bool:
        return self.bucket.object_exists(key=key)

    def get_object(self, key: str) -> bytes:
        if self.bucket.object_exists(key):
            logger.info("file exists")
            return self.bucket.get_object(key)
        else:
            logger.info("file does not exist")
            return None

    def put_object(self, key: str, data: bytes, headers=None) -> None:
        self.bucket.put_object(key, data, headers=headers)

    def put_object_if_not_exists(
        self, data: bytes, file_ext: str, headers=None, path_prefix=None
    ):
        key = hashlib.md5(data).hexdigest()

        if not headers:
            headers = dict()

        if path_prefix:
            key = path_prefix + key
        key += file_ext

        if not self.exists(key):
            self.bucket.put_object(key, data, headers=headers)

        return f"{self.base_url}{key}"

    def _make_url(self):
        base_endpoint = (
            self.endpoint.replace("https://", "").replace("http://", "").strip("/")
        )
        return f"https://{self.bucket_name}.{base_endpoint}/"
