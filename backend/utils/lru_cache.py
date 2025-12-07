import time
import threading
from collections import OrderedDict
from typing import Any, Callable, Optional

from loguru import logger

DEFAULT_CACHE_CAPACITY = 2000
DEFAULT_EXPIRATION_TIME = 60 * 60 * 24


# Thread-safe and expiring LRU cache
class LruCache:
    def __init__(
        self,
        max_size=DEFAULT_CACHE_CAPACITY,
        on_delete_func: Optional[Callable[[Any], Any]] = None,
    ):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.on_delete_func = on_delete_func

    def _call_on_delete(self, value):
        if self.on_delete_func is None or value is None:
            return
        try:
            self.on_delete_func(value)
        except Exception as exc:
            logger.warning(f"Error running cache on_delete_func: {exc}")

    def get(self, key):
        evicted_value = None
        with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            value, valid_ts = entry
            if time.time() > valid_ts:
                entry = self.cache.pop(key, None)
                if entry:
                    evicted_value = entry[0]
            else:
                self.cache.move_to_end(key)
                return value
        self._call_on_delete(evicted_value)
        return None

    def delete(self, key):
        evicted_value = None
        with self.lock:
            entry = self.cache.pop(key, None)
        if entry:
            evicted_value = entry[0]
        self._call_on_delete(evicted_value)

    def put(self, key, value, ttl=DEFAULT_EXPIRATION_TIME):
        evicted_value = None
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    _, evicted_entry = self.cache.popitem(last=False)
                    evicted_value = evicted_entry[0]
            self.cache[key] = (value, time.time() + ttl)
        self._call_on_delete(evicted_value)

    def put_if_not_exists(self, key, value, ttl=DEFAULT_EXPIRATION_TIME):
        logger.info(
            f"Writing {key}:{value} to LRU cache with {ttl} seconds, cache_size={len(self.cache)}."
        )
        evicted_value = None
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return False
            else:
                if len(self.cache) >= self.max_size:
                    _, evicted_entry = self.cache.popitem(last=False)
                    evicted_value = evicted_entry[0]
                self.cache[key] = (value, time.time() + ttl)
                success = True
        self._call_on_delete(evicted_value)
        return success

    def __contains__(self, key):
        with self.lock:
            return key in self.cache and not self.expired(key)

    def expired(self, key):
        entry = self.cache.get(key)
        if entry is None:
            return True
        _, valid_ts = entry
        return time.time() > valid_ts

    def clear_expired(self):
        """Remove all expired items"""
        with self.lock:
            now = time.time()
            keys_to_remove = [
                k for k, (_, valid_ts) in self.cache.items() if now > valid_ts
            ]
        for key in keys_to_remove:
            self.delete(key)

    def size(self):
        with self.lock:
            return len(self.cache)

    def clear(self):
        """Remove all entries and release their resources."""
        values_to_release = []
        with self.lock:
            values_to_release = [entry[0] for entry in self.cache.values()]
            self.cache.clear()
        for value in values_to_release:
            self._call_on_delete(value)
        logger.info(f"Successfully cleared cache, released {len(values_to_release)} value instances")


lru_cache = LruCache()
