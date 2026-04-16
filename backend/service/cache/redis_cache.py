from aiocache.backends.redis import RedisCache
from aiocache import SimpleMemoryCache
from aiocache.base import BaseCache
from aiocache.serializers import JsonSerializer
from db.redis_conn import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_PASSWORD,
    REDIS_DB,
    REDIS_CLUSTER_MODE,
    REDIS_CLUSTER_NODES,
    REDIS_SSL,
)
import os
from loguru import logger
from typing import Any


class RedisClusterCache(BaseCache):
    """
    aiocache-compatible cache backend for Redis Cluster.

    Uses redis.asyncio.cluster.RedisCluster for cluster support.
    """

    def __init__(
        self,
        startup_nodes: list = None,
        password: str = None,
        ssl: bool = False,
        namespace: str = None,
        timeout: int = 15,
        ttl: int = None,
        **kwargs,
    ):
        super().__init__(namespace=namespace, ttl=ttl, **kwargs)
        self.startup_nodes = startup_nodes or []
        self.password = password
        self.ssl = ssl
        self.timeout = timeout
        self._client = None
        self.serializer = JsonSerializer()

    async def _get_client(self):
        """Lazy initialization of Redis Cluster client."""
        if self._client is None:
            try:
                from redis.asyncio.cluster import RedisCluster, ClusterNode
            except ImportError:
                raise ImportError(
                    "redis>=4.1.0 is required for Redis Cluster support. "
                    "Install with: pip install 'redis>=4.1.0'"
                )

            nodes = [
                ClusterNode(host=host, port=port)
                for host, port in self.startup_nodes
            ]
            self._client = RedisCluster(
                startup_nodes=nodes,
                password=self.password,
                ssl=self.ssl,
                socket_timeout=self.timeout,
                decode_responses=False,  # We handle serialization ourselves
            )
        return self._client

    def _build_key(self, key: str, namespace: str = None) -> str:
        """Build the full key with namespace prefix."""
        ns = namespace if namespace is not None else self.namespace
        if ns:
            return f"{ns}:{key}"
        return key

    async def _get(self, key: str, **kwargs) -> Any:
        client = await self._get_client()
        full_key = self._build_key(key)
        value = await client.get(full_key)
        if value is None:
            return None
        return self.serializer.loads(value)

    async def _set(self, key: str, value: Any, ttl: int = None, **kwargs) -> bool:
        client = await self._get_client()
        full_key = self._build_key(key)
        ttl = ttl or self.ttl
        serialized = self.serializer.dumps(value)
        if ttl:
            # Redis SETEX requires integer TTL
            await client.setex(full_key, int(ttl), serialized)
        else:
            await client.set(full_key, serialized)
        return True

    async def _delete(self, key: str, **kwargs) -> int:
        client = await self._get_client()
        full_key = self._build_key(key)
        return await client.delete(full_key)

    async def _exists(self, key: str, **kwargs) -> bool:
        client = await self._get_client()
        full_key = self._build_key(key)
        return await client.exists(full_key) > 0

    async def _clear(self, namespace: str = None, **kwargs) -> bool:
        """Clear is not fully supported in cluster mode due to key distribution."""
        logger.warning(
            "clear() is not fully supported in Redis Cluster mode. "
            "Keys are distributed across nodes."
        )
        return False

    async def close(self) -> None:
        """Close the Redis Cluster connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None


class CacheManager:
    def __init__(self):
        self._cache = None

    def get_cache(self) -> BaseCache:
        # Check if cache is already initialized
        if self._cache is not None:
            # Crucial Check: Ensure the internal client hasn't been
            # detached from the current loop (esp. in tests)
            return self._cache

        # Lazy initialization happens here, inside the active loop
        if os.getenv("DISABLE_REDIS_CACHE_IN_TESTS", "false").lower() == "true":
            logger.info("Using SimpleMemory cache.")
            self._cache = SimpleMemoryCache()
        elif REDIS_CLUSTER_MODE:
            logger.info(
                f"Connecting to Redis Cluster with nodes: {REDIS_CLUSTER_NODES}"
            )
            self._cache = RedisClusterCache(
                startup_nodes=REDIS_CLUSTER_NODES,
                password=REDIS_PASSWORD,
                ssl=REDIS_SSL,
                namespace="pairag",
                timeout=15,
                ttl=60 * 60 * 24,
            )
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

def kb_metadata_schema_key(tenant_id: str, kb_id: str) -> str:
    return f"tenant:{tenant_id}:kb_metadata_schema:{kb_id}"

def vector_table_name_key(tenant_id: str, kb_id: str) -> str:
    return f"tenant:{tenant_id}:kb_id:{kb_id}:vector_table_name"
