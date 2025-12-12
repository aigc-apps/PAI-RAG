from io import BytesIO
import os
from typing import BinaryIO, Optional
from pairag.file.store.base import BaseFileStore, FileUploadResult
import aiofiles
from loguru import logger
import traceback
import time
import hmac
import hashlib
import base64
from urllib.parse import urlencode


SECRET_KEY = os.environ.get("LOCAL_FILE_STORE_SECRET_KEY", "your-very-strong-secret-key").encode("utf-8")


def generate_signed_url(file_path: str, base_url: str, ttl_seconds: int = 3600) -> str:
    """
    生成带签名和有效期的文件下载链接
    :param file_path: 文件路径
    :param base_url: 你提供下载接口的基础 URL，比如 "/v1/fileview"
    :param ttl_seconds: 链接有效时间（秒）
    """
    expires = int(time.time()) + ttl_seconds  # 过期时间的时间戳（秒）

    # 构造参与签名的字符串（顺序要固定）
    data_to_sign = f"{file_path}:{expires}".encode("utf-8")

    # HMAC-SHA256 签名
    signature_bytes = hmac.new(SECRET_KEY, data_to_sign, hashlib.sha256).digest()

    # 为了放到 URL 里，做 URL-safe 的 base64
    signature = base64.urlsafe_b64encode(signature_bytes).decode("utf-8").rstrip("=")

    # 把参数放到 query 里
    query = urlencode({"file_path": file_path, "expires": expires, "sig": signature})

    return f"{base_url}?{query}"



class LocalFileStore(BaseFileStore):
    def __init__(self, base_path: str):
        super().__init__()

        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
        elif not os.path.isdir(base_path):
            raise ValueError("base_path must be a directory")

        self.base_path = base_path
    
    def get_url(self, file_path: str, tenant_id: str) -> str:
        return generate_signed_url(file_path, "/v1/fileview", 3600)

    def write(self, file: BinaryIO, file_name: str, file_path: str, tenant_id: str) -> FileUploadResult:
        try:
            full_path = os.path.join(self.base_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, "wb") as f:
                f.write(file.read())
            logger.info(f"Saved file to {full_path}.")
            return FileUploadResult(
                file_name=file_name,
                file_path=file_path,
            )
        except Exception as e:
            logger.error(f"Failed to write file {file_path}. error: {traceback.format_exc()}")
            raise

    def read(self, file_path: str, tenant_id: str) -> Optional[BinaryIO]:
        try:
            full_path = os.path.join(self.base_path, file_path)
            with open(full_path, "rb") as f:
                return BytesIO(f.read())
        except Exception as e:
            logger.error(f"Failed to read file {file_path}. error: {traceback.format_exc()}")
            raise

    async def get_url_async(self, file_path: str, tenant_id: str) -> Optional[str]:
        return generate_signed_url(file_path, "/v1/fileview", 3600)

    async def write_async(self, file: BinaryIO, file_name: str, file_path: str, tenant_id: str) -> FileUploadResult:
        try:
            full_path = os.path.join(self.base_path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            async with aiofiles.open(full_path, "wb") as f:
                await f.write(file.read())
            logger.info(f"Saved file to {full_path}.")
            return FileUploadResult(
                file_name=file_name,
                file_path=file_path,
            )
        except Exception as e:
            logger.error(f"Failed to write file {file_path}. error: {traceback.format_exc()}")
            raise

    async def read_async(self, file_path: str, tenant_id: str) -> Optional[BinaryIO]:
        try:
            full_path = os.path.join(self.base_path, file_path)
            async with aiofiles.open(full_path, "rb") as f:
                return BytesIO(await f.read())
        except Exception as e:
            logger.error(f"Failed to read file {file_path}. error: {traceback.format_exc()}")
            raise
