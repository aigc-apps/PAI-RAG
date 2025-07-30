import time
import threading
from collections import OrderedDict
from loguru import logger

DEFAULT_CACHE_CAPACITY = 2000
DEFAULT_EXPIRATION_TIME = 60 * 60 * 24


# Thread-safe and expiring LRU cache
class LruCache:
    def __init__(self, maxsize=DEFAULT_CACHE_CAPACITY):
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            value, valid_ts = entry
            if time.time() > valid_ts:
                self.cache.pop(key)
                return None
            # Mark as recently used
            self.cache.move_to_end(key)
            return value

    def delete(self, key):
        self.cache.pop(key, None)

    def put(self, key, value, ttl=DEFAULT_EXPIRATION_TIME):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            elif len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)  # Remove least recently used item
            self.cache[key] = (value, time.time() + ttl)

    def put_if_not_exists(self, key, value, ttl=DEFAULT_EXPIRATION_TIME):
        logger.info(
            f"Writing {key}:{value} to LRU cache with {ttl} seconds, cache_size={len(self.cache)}."
        )
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return False
            else:
                if len(self.cache) >= self.maxsize:
                    self.cache.popitem(last=False)  # Remove least recently used item
                self.cache[key] = (value, time.time() + ttl)
                return True

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
                self.cache.pop(key, None)

    def size(self):
        with self.lock:
            return len(self.cache)


lru_cache = LruCache()
