from io import BytesIO
import os
import oss2
from alibabacloud_credentials import providers
from oss2.credentials import EnvironmentVariableCredentialsProvider, CredentialsProvider
from typing import BinaryIO, Optional
from pairag.file.store.base import BaseFileStore, FileUploadResult
from loguru import logger
from oss2.models import BucketCors, CorsRule
import traceback
import asyncio

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
            if os.getenv('OSS_ACCESS_KEY_ID') and os.getenv('OSS_ACCESS_KEY_SECRET'):
                credentials_provider = EnvironmentVariableCredentialsProvider()
            else:
                # 获取EAS ram role
                credentials_provider = providers.DefaultCredentialsProvider()

        auth = oss2.ProviderAuth(credentials_provider)
        self.bucket = oss2.Bucket(auth=auth, endpoint=endpoint, bucket_name=bucket)
        self.endpoint = endpoint
        # 判断是否为内网地址，如果是则生成对应的公网地址用于签名 URL
        self.is_internal = "-internal" in endpoint.lower()
        self.public_endpoint = endpoint.replace("-internal", "").replace("-Internal", "") if self.is_internal else endpoint
        rule = CorsRule(
            allowed_origins=["*"],
            allowed_methods=["GET", "HEAD", "POST", "PUT", "DELETE"],
            allowed_headers=["*"],
            max_age_seconds=1000,
        )
        self.prefix_path = prefix_path
        try:
            self.bucket.put_bucket_cors(BucketCors([rule]))
        except Exception as ex:
            logger.warning(f"Failed to set CORS for bucket {bucket}. error: {ex}")
            pass

    def get_url(self, file_path: str, tenant_id: str) -> Optional[str]:
        try:
            oss_file_key = os.path.join(self.prefix_path, file_path)
            oss_url = self.bucket.sign_url("GET", oss_file_key, 3600)
            # 如果是内网地址，替换为公网地址以便外部访问
            if self.is_internal:
                oss_url = oss_url.replace(self.endpoint, self.public_endpoint)
            logger.info(f"Get url {oss_url} for file {file_path}.")
            return oss_url
        except Exception as e:
            logger.error(f"Failed to get url for file {file_path}. error: {traceback.format_exc()}")
            raise
    
    def write(self, file: BinaryIO, file_name: str, file_path: str, tenant_id: str) -> FileUploadResult:
        try:
            oss_file_key = os.path.join(self.prefix_path, file_path)
            self.bucket.put_object(key=oss_file_key, data=file.read())
            logger.info(f"Saved oss file {file_name} to {oss_file_key}.")
            return FileUploadResult(
                file_name=file_name,
                file_path=file_path,
            )
        except Exception as e:
            logger.error(f"Failed to write file {file_path}. error: {traceback.format_exc()}")
            raise

    def read(self, file_path: str, tenant_id: str) -> Optional[BinaryIO]:
        oss_file_key = os.path.join(self.prefix_path, file_path)
        try:
            object_result = self.bucket.get_object(key=oss_file_key)
            return BytesIO(object_result.read())
        except Exception as e:
            logger.error(f"Failed to read file {file_path}. error: {traceback.format_exc()}")
            raise

    async def get_url_async(self, file_path: str, tenant_id: str) -> Optional[str]:
        try:
            oss_file_key = os.path.join(self.prefix_path, file_path)
            oss_url = self.bucket.sign_url("GET", oss_file_key, 3600)
            # 如果是内网地址，替换为公网地址以便外部访问
            if self.is_internal:
                oss_url = oss_url.replace(self.endpoint, self.public_endpoint)
            logger.info(f"Get url {oss_url} for file {file_path}.")
            return oss_url
        except Exception as e:
            logger.error(f"Failed to get url for file {file_path}. error: {traceback.format_exc()}")
            raise

    async def write_async(self, file: BinaryIO, file_name: str, file_path: str, tenant_id: str) -> FileUploadResult:
        oss_file_key = os.path.join(self.prefix_path, file_path)
        try:
            write_task = asyncio.to_thread(self.bucket.put_object, key=oss_file_key, data=file.read())
            await write_task
            logger.info(f"Saved oss file {file_path} to {oss_file_key}.")

            return FileUploadResult(
                file_name=file_name,
                file_path=file_path,
            )
        except Exception as e:
            logger.error(f"Failed to write file {file_path}. error: {traceback.format_exc()}")
            raise

    async def read_async(self, file_path: str, tenant_id: str) -> Optional[BinaryIO]:
        oss_file_key = os.path.join(self.prefix_path, file_path)
        try:
            read_task = asyncio.to_thread(self.bucket.get_object, key=oss_file_key)
            object_result = await read_task
            return BytesIO(object_result.read())
        except Exception as e:
            logger.error(f"Failed to read file {file_path}. error: {traceback.format_exc()}")
            raise
