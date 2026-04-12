from abc import ABC, abstractmethod
from typing import Optional, BinaryIO
from pydantic import BaseModel
import asyncio

class FileUploadResult(BaseModel):
    file_name: str
    file_path: str


class BaseFileStore(ABC):
    """
    抽象基类，定义统一的文件操作接口。
    """
    def write(self, file: BinaryIO, file_path: str, tenant_id: str) -> FileUploadResult:
        try:
            return asyncio.run(self.write_async(file=file, file_path=file_path, tenant_id=tenant_id))
        except RuntimeError as e:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.write_async(file=file, file_path=file_path, tenant_id=tenant_id))
    
    def read(self, file_path: str, tenant_id: str) -> Optional[BinaryIO]:
        try:
            return asyncio.run(self.read_async(file_path=file_path, tenant_id=tenant_id))
        except RuntimeError as e:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.read_async(file_path=file_path, tenant_id=tenant_id))
    
    def get_url(self, file_path: str, tenant_id: str) -> Optional[str]:
        try:
            return asyncio.run(self.get_url_async(file_path=file_path, tenant_id=tenant_id))
        except RuntimeError as e:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.get_url_async(file_path=file_path, tenant_id=tenant_id))

    @abstractmethod
    async def write_async(self, file: BinaryIO, file_path: str, tenant_id: str) -> FileUploadResult:
        """
        将文件内容保存到指定路径。

        :param file: 文件对象或二进制流
        :param file_path: 存储路径（例如: 'folder/file.txt'）
        """
        pass

    @abstractmethod
    async def read_async(self, file_path: str, tenant_id: str) -> Optional[BinaryIO]:
        """
        从指定路径加载文件内容。

        :param file_path: 文件路径
        :return: 返回一个文件对象或字节流
        """
        pass


    @abstractmethod
    async def get_url_async(self, file_path: str, tenant_id: str) -> Optional[str]:
        """
        获取指定路径的文件URL。

        :param file_path: 文件路径
        :return: 文件URL
        """
        pass

    async def cleanup(self):
        """关闭底层连接资源。子类可覆盖。"""
        pass
