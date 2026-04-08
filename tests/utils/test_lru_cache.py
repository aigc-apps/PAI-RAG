import sys
import os
import time
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from utils.lru_cache import LruCache


class TestLruCache:
    def test_basic_put_get(self):
        cache = LruCache(max_size=10)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_key(self):
        cache = LruCache(max_size=10)
        assert cache.get("nonexistent") is None

    def test_delete(self):
        cache = LruCache(max_size=10)
        cache.put("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_ttl_expiry(self):
        cache = LruCache(max_size=10)
        base_time = 1000.0
        with patch("utils.lru_cache.time.time", return_value=base_time):
            cache.put("key1", "value1", ttl=10)
        # Before expiry
        with patch("utils.lru_cache.time.time", return_value=base_time + 5):
            assert cache.get("key1") == "value1"
        # After expiry
        with patch("utils.lru_cache.time.time", return_value=base_time + 11):
            assert cache.get("key1") is None

    def test_lru_eviction(self):
        cache = LruCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # Should evict "a" (oldest)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_lru_access_order(self):
        cache = LruCache(max_size=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # Access "a" making "b" the LRU
        cache.put("c", 3)  # Should evict "b"
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_put_if_not_exists_new_key(self):
        cache = LruCache(max_size=10)
        result = cache.put_if_not_exists("key1", "value1")
        assert result is True
        assert cache.get("key1") == "value1"

    def test_put_if_not_exists_existing_key(self):
        cache = LruCache(max_size=10)
        cache.put("key1", "value1")
        result = cache.put_if_not_exists("key1", "value2")
        assert result is False
        assert cache.get("key1") == "value1"  # Original value preserved

    def test_on_delete_func_callback(self):
        deleted_values = []
        cache = LruCache(max_size=2, on_delete_func=lambda v: deleted_values.append(v))
        cache.put("a", "val_a")
        cache.put("b", "val_b")
        cache.put("c", "val_c")  # Evicts "a"
        assert "val_a" in deleted_values

    def test_on_delete_func_on_explicit_delete(self):
        deleted_values = []
        cache = LruCache(max_size=10, on_delete_func=lambda v: deleted_values.append(v))
        cache.put("key1", "value1")
        cache.delete("key1")
        assert "value1" in deleted_values

    def test_clear(self):
        deleted_values = []
        cache = LruCache(max_size=10, on_delete_func=lambda v: deleted_values.append(v))
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.size() == 0
        assert cache.get("a") is None
        assert len(deleted_values) == 2

    def test_clear_expired(self):
        cache = LruCache(max_size=10)
        base_time = 1000.0
        with patch("utils.lru_cache.time.time", return_value=base_time):
            cache.put("short_ttl", "val1", ttl=5)
            cache.put("long_ttl", "val2", ttl=100)

        with patch("utils.lru_cache.time.time", return_value=base_time + 10):
            cache.clear_expired()
            assert cache.get("long_ttl") == "val2"
            # short_ttl should have been removed
            assert cache.size() == 1

    def test_size(self):
        cache = LruCache(max_size=10)
        assert cache.size() == 0
        cache.put("a", 1)
        assert cache.size() == 1
        cache.put("b", 2)
        assert cache.size() == 2

    def test_contains(self):
        cache = LruCache(max_size=10)
        cache.put("key1", "value1")
        assert "key1" in cache
        assert "key2" not in cache

    def test_put_overwrites_existing(self):
        cache = LruCache(max_size=10)
        cache.put("key1", "value1")
        cache.put("key1", "value2")
        assert cache.get("key1") == "value2"
