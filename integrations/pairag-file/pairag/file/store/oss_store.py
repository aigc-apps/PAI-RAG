from io import BytesIO
import os
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider, CredentialsProvider
from typing import BinaryIO, Optional
from pairag.file.store.base import BaseFileStore
from loguru import logger
from oss2.models import BucketCors, CorsRule

DEFAULT_OSS_PREFIX = "pairag_knowledgebases"


class OssFileStore(BaseFileStore):
    def __init__(
        self,
        bucket: str,
        endpoint: str,
        prefix_path: str = DEFAULT_OSS_PREFIX,
        credentials_provider: Optional[CredentialsProvider] = None,
    ):
        super().__init__()
        if credentials_provider is None:
            credentials_provider = EnvironmentVariableCredentialsProvider()

        auth = oss2.ProviderAuth(credentials_provider)
        self.bucket = oss2.Bucket(auth=auth, endpoint=endpoint, bucket_name=bucket)
        rule = CorsRule(
            allowed_origins=["*"],
            allowed_methods=["GET", "HEAD"],
            allowed_headers=["*"],
            max_age_seconds=1000,
        )
        try:
            self.bucket.put_bucket_cors(BucketCors([rule]))
            self.prefix_path = prefix_path
        except Exception as ex:
            logger.warning(f"Failed to set CORS for bucket {bucket}. error: {ex}")
            pass


    def get_url(self, file_path: str):
        oss_file_key = os.path.join(self.prefix_path, file_path)
        oss_url = self.bucket.sign_url("GET", oss_file_key, 3600)
        logger.info(f"Get url {oss_url} for file {file_path}.")
        return oss_url

    def save(self, file: BinaryIO, file_path: str) -> None:
        oss_file_key = os.path.join(self.prefix_path, file_path)
        self.bucket.put_object(key=oss_file_key, data=file.read())
        logger.info(f"Saved oss file {file_path} to {oss_file_key}.")

    def load(self, file_path: str) -> Optional[BinaryIO]:
        oss_file_key = os.path.join(self.prefix_path, file_path)
        object_result = self.bucket.get_object(oss_file_key)
        return BytesIO(object_result.read())

    def exists(self, file_path: str) -> bool:
        oss_file_key = os.path.join(self.prefix_path, file_path)
        return self.bucket.object_exists(oss_file_key)
