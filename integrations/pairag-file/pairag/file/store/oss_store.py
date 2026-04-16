from io import BytesIO
import os
from typing import BinaryIO, Optional, List
from pairag.file.store.base import BaseFileStore, FileUploadResult
from loguru import logger
import traceback
import alibabacloud_oss_v2 as oss
from alibabacloud_oss_v2.credentials import EnvironmentVariableCredentialsProvider, CredentialsProvider
import alibabacloud_oss_v2.aio as oss_aio
from datetime import datetime, timedelta, timezone
from pairag.file.utils.oss_utils import get_region_from_endpoint

DEFAULT_OSS_PREFIX = "pairag_knowledgebases"
DEFAULT_SIGN_EXPIRE_HOURS = 72


class OssFileStore(BaseFileStore):
    def __init__(
        self,
        bucket: str,
        endpoint: str,
        prefix_path: str = DEFAULT_OSS_PREFIX,
        credentials_provider: Optional[CredentialsProvider] = None,
    ):
        super().__init__()
        self.bucket = bucket
        self.is_internal = "-internal" in endpoint.lower()
        self.endpoint = endpoint
        self.region = get_region_from_endpoint(endpoint)
        self.public_endpoint = endpoint.replace("-internal", "").replace("-Internal", "") if self.is_internal else endpoint
        self.prefix_path = prefix_path

        if credentials_provider is None:
            if os.getenv('OSS_ACCESS_KEY_ID') and os.getenv('OSS_ACCESS_KEY_SECRET'):
                credentials_provider = EnvironmentVariableCredentialsProvider()
            else:
                # 获取EAS ram role
                from alibabacloud_credentials import providers as credential_providers
                credentials_provider = credential_providers.DefaultCredentialsProvider()

        cfg = oss.config.load_default()
        cfg.region = self.region
        cfg.endpoint = endpoint
        cfg.credentials_provider = credentials_provider

        self.async_client = oss_aio.AsyncClient(cfg)
        self.client = oss.Client(cfg)

        cors_rule = oss.CORSRule(
            allowed_origins=["*"],
            allowed_methods=["GET", "HEAD", "POST", "PUT", "DELETE"],
            allowed_headers=["*"],
            max_age_seconds=1000,
        )
        try:
            self.client.put_bucket_cors(
                oss.PutBucketCorsRequest(
                    bucket=self.bucket,
                    cors_configuration=oss.CORSConfiguration(
                        cors_rules=[cors_rule]
                    )
                )
            )
        except Exception as ex:
            logger.warning(f"Failed to set CORS for bucket {bucket}. error: {ex}")

    def _get_sign_expire_time(self) -> datetime:
        return datetime.now(timezone.utc) + timedelta(hours=DEFAULT_SIGN_EXPIRE_HOURS)

    def get_url(self, file_path: str, tenant_id: str) -> Optional[str]:
        try:
            oss_file_key = os.path.join(self.prefix_path, file_path)
            presign_result = self.client.presign(
                request=oss.GetObjectRequest(
                    bucket=self.bucket,
                    key=oss_file_key
                ),
                expiration=self._get_sign_expire_time()
            )
            oss_url = presign_result.url

            if self.is_internal:
                oss_url = oss_url.replace(self.endpoint, self.public_endpoint)
            logger.info(f"Get url for file {file_path}.")
            return oss_url
        except Exception as e:
            logger.error(f"Failed to get url for file {file_path}. error: {traceback.format_exc()}")
            raise

    def write(self, file: BinaryIO, file_name: str, file_path: str, tenant_id: str) -> FileUploadResult:
        try:
            oss_file_key = os.path.join(self.prefix_path, file_path)
            file.seek(0)
            file_data = file.read()
            result = self.client.put_object(
                oss.PutObjectRequest(
                    bucket=self.bucket,
                    key=oss_file_key,
                    body=file_data
                )
            )
            logger.info(f"Saved oss file {file_name} to {oss_file_key}. status_code: {result.status_code}")

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
            result = self.client.get_object(
                oss.GetObjectRequest(
                    bucket=self.bucket,
                    key=oss_file_key
                )
            )
            logger.info(f"Read oss file {file_path} from {oss_file_key}. status_code: {result.status_code}")
            return BytesIO(result.body.content)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}. error: {traceback.format_exc()}")
            raise

    async def get_url_async(self, file_path: str, tenant_id: str) -> Optional[str]:
        # Presigning is a local crypto operation (no network I/O),
        # so reuse the sync client to avoid unnecessary HTTP requests.
        return self.get_url(file_path, tenant_id)

    async def write_async(self, file: BinaryIO, file_name: str, file_path: str, tenant_id: str) -> FileUploadResult:
        oss_file_key = os.path.join(self.prefix_path, file_path)
        try:
            file.seek(0)
            file_data = file.read()
            result = await self.async_client.put_object(
                oss.PutObjectRequest(
                    bucket=self.bucket,
                    key=oss_file_key,
                    body=file_data
                )
            )
            logger.info(f"Saved oss file {file_path} to {oss_file_key}. status_code: {result.status_code}")

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
            result = await self.async_client.get_object(
                oss.GetObjectRequest(
                    bucket=self.bucket,
                    key=oss_file_key
                )
            )
            status_code = result.status_code
            logger.info(f"Read oss file {file_path} from {oss_file_key}. status_code: {status_code}")
            return BytesIO(result.body.content)
        except Exception as e:
            logger.error(f"Failed to read file {file_path}. error: {traceback.format_exc()}")
            raise

    async def list_objects_async(
        self,
        prefix: str,
        tenant_id: Optional[str] = None,
    ) -> List[oss.ObjectProperties]:
        try:
            continuation_token = None
            page_index = 0
            while True:
                page_index += 1
                result = await self.async_client.list_objects_v2(
                    oss.ListObjectsV2Request(
                        bucket=self.bucket,
                        max_keys=1000,
                        prefix=prefix,
                        continuation_token=continuation_token,
                    )
                )

                files = result.contents or []
                logger.info(f"List objects for prefix {prefix}. page {page_index}. file count: {len(files)}, next_continuation_token: {result.next_continuation_token}")
                yield files

                if result.next_continuation_token:
                    continuation_token = result.next_continuation_token
                else:
                    break
        except Exception as e:
            logger.error(f"Failed to list objects for prefix {prefix}. error: {traceback.format_exc()}")
            raise

    async def cleanup(self):
        await self.async_client.close()
