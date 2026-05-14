"""Runtime-mutable config persisted to disk.

Currently only the active LLM model is mutable at runtime. The active model is
read by every fresh ``LLMClient`` construction across the HTTP server, the
Celery worker, the ACP server and the background memory reviewer; persisting
to a JSON file keeps all those processes in sync without a shared in-memory
state. ``mtime``-based caching keeps the read path cheap.
"""
import json
import os
import tempfile
import threading
from pathlib import Path

import settings as config

ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT / 'memory'
ACTIVE_MODEL_PATH = RUNTIME_DIR / 'active_model.json'

_LOCK = threading.RLock()
_CACHE = {'mtime': 0.0, 'value': None}


def validate_model_name(name):
    if not isinstance(name, str):
        raise ValueError('model must be a string')
    s = name.strip()
    if not s:
        raise ValueError('model must be non-empty')
    if len(s) > 200:
        raise ValueError('model name too long (max 200 chars)')
    if any(ch.isspace() for ch in s):
        raise ValueError('model must not contain whitespace')
    return s


def get_active_model():
    fallback = getattr(config, 'MODEL', 'qwen-plus')
    with _LOCK:
        try:
            mtime = ACTIVE_MODEL_PATH.stat().st_mtime
        except FileNotFoundError:
            return fallback
        if mtime != _CACHE['mtime']:
            try:
                with ACTIVE_MODEL_PATH.open('r', encoding='utf-8') as f:
                    data = json.load(f)
            except (OSError, ValueError):
                return fallback
            value = (data.get('model') if isinstance(data, dict) else None) or ''
            value = value.strip() or None
            _CACHE.update(value=value, mtime=mtime)
        return _CACHE['value'] or fallback


def set_active_model(name):
    s = validate_model_name(name)
    with _LOCK:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='active_model.', suffix='.json', dir=str(RUNTIME_DIR))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump({'model': s}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, ACTIVE_MODEL_PATH)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
        _CACHE.update(value=s, mtime=ACTIVE_MODEL_PATH.stat().st_mtime)
        return s


def reset_cache():
    """Clear the in-process cache. Tests use this; production code shouldn't need it."""
    with _LOCK:
        _CACHE.update(value=None, mtime=0.0)
