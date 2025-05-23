import hashlib
import pathlib
import oss2
import os
from oss2.credentials import EnvironmentVariableCredentialsProvider
from pai_rag.file.readers.pai.constants import ACCEPTABLE_DOC_TYPES
from loguru import logger


class PaiOssStore:
    def __init__(self, bucket_name: str, endpoint: str, oss_bucket: oss2.Bucket = None):
        self.bucket_name = bucket_name
        self.endpoint = endpoint
        self.base_url = self._make_url()

        if oss_bucket is None:
            auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
            self.bucket = oss2.Bucket(auth, endpoint, bucket_name)
        else:
            self.bucket = oss_bucket

        logger.info(
            f"Initializing OSS store with bucket: {bucket_name} with endpoint {self.endpoint}."
        )

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
            key = os.path.join(path_prefix, key)
        key += file_ext

        if not self.exists(key):
            self.bucket.put_object(key, data, headers=headers)

        return f"{self.base_url}{key}"

    def get_obj_key_url(self, file_path: str):
        return f"{self.base_url}{file_path}"

    def _make_url(self):
        base_endpoint = (
            self.endpoint.replace("https://", "").replace("http://", "").strip("/")
        )
        return f"https://{self.bucket_name}.{base_endpoint}/"

    def list_objects(self, prefix: str):
        """
        列出存储桶中指定前缀的对象列表。


        参数:
        - prefix (str): 对象名的前缀，用于筛选满足条件的对象。默认为空字符串，表示不指定前缀，即列出所有对象。

        返回:
        - list: 包含满足前缀条件的所有对象的列表。
        """
        object_list = []
        for obj in oss2.ObjectIteratorV2(self.bucket, prefix):
            logger.info("iterating oss file: " + obj.key)
            object_list.append(obj)
        return object_list

    def put_object_acl(self, key, permission):
        if key.endswith(".txt"):
            self.bucket.update_object_meta(
                key, {"Content-Type": "text/plain;charset=utf-8"}
            )

        res = self.bucket.put_object_acl(key=key, permission=permission)

        return res.status == 200

    def parse_oss_prefix(self, oss_path):
        oss_path = oss_path.strip()
        assert oss_path.startswith(
            f"oss://{self.bucket_name}/"
        ), f"OSS path should start with oss://{self.bucket_name}/"
        oss_prefix = oss_path[len(f"oss://{self.bucket_name}/") :]
        assert len(oss_prefix) > 0, "Must provide a oss prefix to download data."

        return oss_prefix

    def download_oss_file_to_local(
        self,
        oss_path: str,
        local_path: str,
        need_set_public: bool = False,
        file_types: set[str] = ACCEPTABLE_DOC_TYPES,
    ):
        files = []
        prefix = self.parse_oss_prefix(oss_path)
        object_list = self.list_objects(prefix)

        base_oss_path = pathlib.Path(oss_path)

        for oss_obj in object_list:
            if (
                not oss_obj.key.endswith("/")
                and pathlib.Path(oss_obj.key).suffix.lower() in file_types
            ):  # 不是目录
                logger.info(f"Downloading oss object: {oss_obj.key}")
                if need_set_public:
                    try:
                        _ = self.put_object_acl(oss_obj.key, "public-read")
                    except Exception:
                        logger.error(
                            f"Failed to set_public access for oss file {oss_obj.key}"
                        )
                        raise

                current_oss_path = pathlib.Path(oss_obj.key)
                relative_path = current_oss_path.relative_to(base_oss_path)

                save_filename = os.path.join(local_path, relative_path)
                save_dir = os.path.dirname(save_filename)
                os.makedirs(save_dir, exist_ok=True)

                self.get_object_to_file(key=oss_obj.key, filename=save_filename)
                files.append(save_filename)
                logger.info(f"Downloaded oss object: {oss_obj.key} to {save_filename}")
            else:
                logger.error(f"Skip load oss document {oss_obj.key}.")

        if not files:
            raise ValueError(f"No file found at OSS path '{oss_path}'.")
        return files
