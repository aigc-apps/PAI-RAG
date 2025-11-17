"""Shared aiohttp ClientSession singleton."""
from typing import Optional
import aiohttp


class HttpSessionShared:
    """Shared aiohttp ClientSession singleton."""

    default: Optional[aiohttp.ClientSession] = None

    @classmethod
    async def ensure_session(cls):
        """Ensure session exists."""
        if cls.default is None or cls.default.closed:
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
