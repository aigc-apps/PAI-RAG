import os
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from typing import BinaryIO, Optional
from pairag.store.file.base import BaseFileStore
from loguru import logger

DEFAULT_OSS_PREFIX = "pairag_docs"


class OssFileStore(BaseFileStore):
    def __init__(
        self, bucket: str, endpoint: str, prefix_path: str = DEFAULT_OSS_PREFIX
    ):
        super().__init__()
        auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
        self.bucket = oss2.Bucket(auth, endpoint, bucket)
        self.prefix_path = prefix_path

        logger.info(
            f"Created oss file store with prefix {prefix_path} bucket {bucket} and endpoint {endpoint}."
        )

    def get_url(self, file_path: str):
        oss_file_key = os.path.join(self.prefix_path, file_path)
        return self.bucket.sign_url("GET", oss_file_key, 3600)

    def save(self, data: BinaryIO, file_path: str) -> None:
        oss_file_key = os.path.join(self.prefix_path, file_path)
        self.bucket.put_object(oss_file_key, data)
        logger.info(f"Saved oss file {file_path} to {oss_file_key}.")

    def load(self, file_path: str) -> Optional[BinaryIO]:
        oss_file_key = os.path.join(self.prefix_path, file_path)
        return self.bucket.get_object(oss_file_key)

    def exists(self, file_path: str) -> bool:
        oss_file_key = os.path.join(self.prefix_path, file_path)
        return self.bucket.object_exists(oss_file_key)
