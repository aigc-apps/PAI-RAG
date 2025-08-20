from functools import wraps
import asyncio
from typing import Callable
from rag.chunk_helper import read_file_from_db
from loguru import logger

class FileExistenceGuard:
    """文件存在性检查守护器"""

    def __init__(self, file_id: str):
        self.file_id = file_id
        self.cancelled = False

    async def check_exists(self) -> bool:
        """检查文件是否存在"""
        if self.cancelled:
            return False
        check_file_entity = await read_file_from_db(file_id=self.file_id)
        exists = check_file_entity is not None
        if not exists:
            self.cancelled = True
            logger.warning(f"[GUARD] File {self.file_id} no longer exists.")
        return exists

def require_file_exists(guard_getter: Callable = None):
    """装饰器：要求文件存在才能执行函数"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            guard = None
            if guard_getter:
                guard = guard_getter(*args, **kwargs)
            else:
                for arg in args:
                    if isinstance(arg, FileExistenceGuard):
                        guard = arg
                        break
                if not guard:
                    for v in kwargs.values():
                        if isinstance(v, FileExistenceGuard):
                            guard = v
                            break

            # 检查文件存在性
            if guard and not await guard.check_exists():
                logger.warning(f"[DECORATOR] Function {func.__name__} cancelled due to missing file.")
                return  # 直接返回，不执行原函数

            # 执行原函数
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator
