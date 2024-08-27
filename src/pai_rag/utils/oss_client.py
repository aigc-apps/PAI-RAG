import logging
import hashlib
import oss2
import os
from oss2.credentials import EnvironmentVariableCredentialsProvider

logger = logging.getLogger(__name__)


class OssClient:
    def __init__(self, bucket_name: str, endpoint: str, prefix: str = None):
        auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
        self.bucket_name = bucket_name
        self.endpoint = endpoint
        self.base_url = self._make_url()
        # 去除prefix可能存在的前后空格，并去除最后的斜杠
        self.prefix = prefix.strip().rstrip("/")

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

    def get_object_to_file(self, key, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.bucket.get_object_to_file(key=key, filename=filename)

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

    def get_obj_key_url(self, filename: str):
        return f"{self.base_url}{self.prefix}/{filename}"

    def _make_url(self):
        base_endpoint = (
            self.endpoint.replace("https://", "").replace("http://", "").strip("/")
        )
        return f"https://{self.bucket_name}.{base_endpoint}/"

    def list_objects(self):
        """
        列出存储桶中指定前缀的对象列表。

        该方法通过调用oss bucket的list_objects函数，查询与给定前缀匹配的所有对象，并返回这些对象的列表。

        参数:
        - prefix (str): 对象名的前缀，用于筛选满足条件的对象。默认为空字符串，表示不指定前缀，即列出所有对象。

        返回:
        - list: 包含满足前缀条件的所有对象的列表。
        """
        # 调用bucket的list_objects方法，传入前缀参数
        res = self.bucket.list_objects(prefix=self.prefix)
        # 返回查询到的对象列表
        return res.object_list

    def put_object_acl(self, key, permission):
        if key.endswith(".txt"):
            self.bucket.update_object_meta(
                key, {"Content-Type": "text/plain;charset=utf-8"}
            )

        res = self.bucket.put_object_acl(key=key, permission=permission)

        return res.status == 200
