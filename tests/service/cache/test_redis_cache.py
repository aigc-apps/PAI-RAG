import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

import pytest
from unittest.mock import patch, MagicMock
from service.cache.redis_cache import CacheManager, kb_key, kb_name_key, vector_table_name_key


class TestCacheKeyFunctions:
    def test_kb_key(self):
        assert kb_key("t1", "kb1") == "tenant:t1:kb_id:kb1"

    def test_kb_name_key(self):
        assert kb_name_key("t1", "my-kb") == "tenant:t1:kb_name:my-kb"

    def test_vector_table_name_key(self):
        assert vector_table_name_key("t1", "kb1") == "tenant:t1:kb_id:kb1:vector_table_name"


class TestCacheManager:
    @patch.dict(os.environ, {"DISABLE_REDIS_CACHE_IN_TESTS": "true"}, clear=False)
    def test_get_cache_memory_mode(self):
        manager = CacheManager()
        cache = manager.get_cache()
        from aiocache import SimpleMemoryCache
        assert isinstance(cache, SimpleMemoryCache)

    @patch.dict(os.environ, {"DISABLE_REDIS_CACHE_IN_TESTS": "true"}, clear=False)
    def test_get_cache_returns_same_instance(self):
        manager = CacheManager()
        cache1 = manager.get_cache()
        cache2 = manager.get_cache()
        assert cache1 is cache2

    def test_cache_manager_initial_state(self):
        manager = CacheManager()
        assert manager._cache is None

    @patch.dict(os.environ, {"DISABLE_REDIS_CACHE_IN_TESTS": "true"}, clear=False)
    async def test_memory_cache_set_and_get(self):
        manager = CacheManager()
        cache = manager.get_cache()
        await cache.set("test_key", {"data": "value"})
        result = await cache.get("test_key")
        assert result == {"data": "value"}

    @patch.dict(os.environ, {"DISABLE_REDIS_CACHE_IN_TESTS": "true"}, clear=False)
    async def test_memory_cache_get_missing(self):
        manager = CacheManager()
        cache = manager.get_cache()
        result = await cache.get("nonexistent")
        assert result is None

    @patch.dict(os.environ, {"DISABLE_REDIS_CACHE_IN_TESTS": "true"}, clear=False)
    async def test_memory_cache_delete(self):
        manager = CacheManager()
        cache = manager.get_cache()
        await cache.set("del_key", "value")
        await cache.delete("del_key")
        result = await cache.get("del_key")
        assert result is None
