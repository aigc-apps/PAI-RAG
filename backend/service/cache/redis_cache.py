from aiocache.backends.redis import RedisCache
from aiocache import SimpleMemoryCache
from db.redis_conn import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB
import os

if os.getenv("DISABLE_REDIS_CACHE", "false") == "true":
    redis_cache = SimpleMemoryCache()
else:
    redis_cache = RedisCache(
        namespace="pairag",
        endpoint=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=REDIS_DB,
        timeout=15,
        ttl=60 * 60 * 24,
    )


def kb_key(tenant_id: str, kb_id: str) -> str:
    return f"tenant:{tenant_id}:kb_id:{kb_id}"

def kb_name_key(tenant_id: str, name: str) -> str:
    return f"tenant:{tenant_id}:kb_name:{name}"
