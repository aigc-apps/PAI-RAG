"""Shared aiohttp ClientSession singleton."""
from typing import Optional
import aiohttp


class HttpSessionShared:
    """Shared aiohttp ClientSession singleton."""

    default: Optional[aiohttp.ClientSession] = None

    @classmethod
    async def ensure_session(cls):
        """Ensure session exists."""
        # 检查是否需要创建新的 session
        need_new_session = False

        if cls.default is None:
            need_new_session = True
        elif cls.default.closed:
            need_new_session = True
        else:
            # 检查 session 绑定的事件循环是否仍然有效
            try:
                loop = getattr(cls.default, '_loop', None)
                if loop is None or loop.is_closed():
                    need_new_session = True
            except (RuntimeError, AttributeError):
                need_new_session = True

        if need_new_session:
            if cls.default and not cls.default.closed:
                try:
                    await cls.default.close()
                except Exception:
                    pass
            cls.default = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )

        return cls.default

    @classmethod
    async def cleanup(cls):
        """Cleanup session."""
        if cls.default and not cls.default.closed:
            await cls.default.close()
            cls.default = None

    @classmethod
    def get_session(cls):
        """Get the current session if it exists, otherwise None."""
        return cls.default
