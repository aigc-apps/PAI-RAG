"""Runtime-mutable config persisted to disk.

Stored in ``memory/runtime.json`` alongside the multi-provider key-pool config
(see ``provider_pool.py``); both modules read the same file so a single hot
reload propagates active-model and key changes together. ``mtime``-based
caching keeps the read path cheap; writes do read-modify-write so the
provider-pool top-level keys (``providers`` / ``default_provider`` /
``cooldown_seconds``) are preserved untouched.

The active model is read by every fresh ``LLMClient`` construction across the
HTTP server, the Celery worker, the ACP server and the background memory
reviewer; persisting to a JSON file keeps all those processes in sync without
a shared in-memory state.
"""
import json
import os
import tempfile
import threading
from pathlib import Path

import settings as config

ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT / 'memory'
RUNTIME_PATH = RUNTIME_DIR / 'runtime.json'

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
            mtime = RUNTIME_PATH.stat().st_mtime
        except FileNotFoundError:
            return fallback
        if mtime != _CACHE['mtime']:
            try:
                with RUNTIME_PATH.open('r', encoding='utf-8') as f:
                    data = json.load(f)
            except (OSError, ValueError):
                return fallback
            value = (data.get('active_model') if isinstance(data, dict) else None) or ''
            value = value.strip() or None
            _CACHE.update(value=value, mtime=mtime)
        return _CACHE['value'] or fallback


def _read_existing():
    """Best-effort load of the current runtime.json so set_active_model can
    do read-modify-write without clobbering provider config."""
    try:
        with RUNTIME_PATH.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def set_active_model(name):
    s = validate_model_name(name)
    with _LOCK:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        merged = _read_existing()
        merged['active_model'] = s
        fd, tmp = tempfile.mkstemp(prefix='runtime.', suffix='.json', dir=str(RUNTIME_DIR))
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(merged, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, RUNTIME_PATH)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
        _CACHE.update(value=s, mtime=RUNTIME_PATH.stat().st_mtime)
        return s


def reset_cache():
    """Clear the in-process cache. Tests use this; production code shouldn't need it."""
    with _LOCK:
        _CACHE.update(value=None, mtime=0.0)
