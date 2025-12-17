from aiocache.backends.redis import RedisCache
from aiocache import SimpleMemoryCache
from db.redis_conn import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB
import os
from loguru import logger


class CacheManager:
    def __init__(self):
        self._cache = None

    def get_cache(self):
        # Check if cache is already initialized
        if self._cache is not None:
            # Crucial Check: Ensure the internal client hasn't been
            # detached from the current loop (esp. in tests)
            return self._cache

        # Lazy initialization happens here, inside the active loop
        if os.getenv("DISABLE_REDIS_CACHE_IN_TESTS", "false").lower() == "true":
            logger.info("Using SimpleMemory cache.")
            self._cache = SimpleMemoryCache()
        else:
            logger.info(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
            self._cache = RedisCache(
                namespace="pairag",
                endpoint=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                db=REDIS_DB,
                timeout=15,
                ttl=60 * 60 * 24,
            )
        return self._cache

# Create a single manager instance globally
cache_manager = CacheManager()


def kb_key(tenant_id: str, kb_id: str) -> str:
    return f"tenant:{tenant_id}:kb_id:{kb_id}"

def kb_name_key(tenant_id: str, name: str) -> str:
    return f"tenant:{tenant_id}:kb_name:{name}"

def vector_table_name_key(tenant_id: str, kb_id: str) -> str:
    return f"tenant:{tenant_id}:kb_id:{kb_id}:vector_table_name"
