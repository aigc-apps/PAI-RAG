from abc import ABC, abstractmethod
from typing import Optional, BinaryIO


class BaseFileStore(ABC):
    """
    抽象基类，定义统一的文件操作接口。
    """

    @abstractmethod
    def save(self, file: BinaryIO, file_path: str) -> None:
        """
        将文件内容保存到指定路径。

        :param file: 文件对象或二进制流
        :param file_path: 存储路径（例如: 'folder/file.txt'）
        """
        pass

    @abstractmethod
    def load(self, file_path: str) -> Optional[BinaryIO]:
        """
        从指定路径加载文件内容。

        :param file_path: 文件路径
        :return: 返回一个文件对象或字节流
        """
        pass

    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """
        检查指定路径是否存在文件。

        :param file_path: 文件路径
        :return: True/False
        """
        pass

    @abstractmethod
    def get_url(self, file_path: str) -> str:
        """
        获取指定路径的文件URL。

        :param file_path: 文件路径
        :return: 文件URL
        """
        pass
