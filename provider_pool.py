"""Multi-provider, multi-key LLM credential pool.

Reads ``memory/runtime.json`` (mtime-cached, hot-reloadable; same file as
``runtime_config.py`` — top-level keys ``providers`` / ``default_provider`` /
``cooldown_seconds`` are owned here, ``active_model`` is owned by
``runtime_config``). When the file is absent or contains no provider entry,
falls back to a single-provider config built from the env ``API_KEY`` /
``API_BASE`` so existing deployments keep behaving like today.

Two responsibilities:

1. Round-robin pick of a live ``(api_key, api_base)`` for a given provider.
2. Failure feedback: 401/403 evicts the key permanently within this process,
   429 cools it down for ``cooldown_seconds`` (default 300s) and revives it
   automatically when the deadline passes. Anything else (including 5xx) is a
   no-op so transient upstream blips don't drain the pool.

Each Python process keeps its own pool state — there is intentionally no
cross-process sync. A bad key gets noticed by every process the first time
*that* process tries it, then evicted; this is acceptable until traffic at
this scale demands a Redis-backed shared pool.
"""
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import settings as config


ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT / 'memory'
RUNTIME_PATH = RUNTIME_DIR / 'runtime.json'

DEFAULT_COOLDOWN_SECONDS = 300
EVICT_STATUS_CODES = frozenset({401, 403})
COOLDOWN_STATUS_CODES = frozenset({429})


class NoLiveKeyError(RuntimeError):
    """Raised when every key for a provider is evicted or in cooldown."""

    def __init__(self, provider):
        super().__init__(f'no live api key for provider {provider!r}')
        self.provider = provider


class UnknownProviderError(KeyError):
    """Raised when a caller asks for a provider that isn't configured."""


@dataclass
class CredBundle:
    provider: str
    key_id: int
    api_key: str
    api_base: str


@dataclass
class _KeyState:
    api_key: str
    evicted: bool = False
    cooldown_until: float = 0.0


@dataclass
class _ProviderState:
    api_base: str
    keys: list = field(default_factory=list)  # list[_KeyState]
    model_prefixes: tuple = ()
    cursor: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


# Module-level singleton.  ``_FILE_LOCK`` guards file-load (mtime check + reload);
# each provider has its own ``lock`` so acquire on provider A doesn't block B.
_FILE_LOCK = threading.RLock()
_STATE = {
    'mtime': -1.0,             # -1 sentinel: nothing loaded yet
    'default_provider': '',
    'cooldown_seconds': DEFAULT_COOLDOWN_SECONDS,
    'providers': {},           # name → _ProviderState
}


# ──────────────────────────── Loading ──────────────────────────── #

def _bootstrap_from_env():
    """Synthesise a single-provider config from settings when no JSON file exists.

    Preserves pre-multi-provider behaviour: one ``qwen`` provider, one key,
    base URL from settings. ``model_prefixes`` is intentionally empty so any
    model name routes here (default-provider fallback in ``resolve_provider``).
    """
    api_key = getattr(config, 'API_KEY', '') or ''
    api_base = getattr(config, 'API_BASE', '') or 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    return {
        'default_provider': 'qwen',
        'cooldown_seconds': DEFAULT_COOLDOWN_SECONDS,
        'providers': {
            'qwen': _ProviderState(
                api_base=api_base,
                keys=[_KeyState(api_key=api_key)] if api_key else [],
                model_prefixes=(),
            ),
        },
    }


def _parse_config(data):
    """Convert raw JSON dict → in-memory ``_ProviderState`` map.

    Tolerates partial / malformed entries: a provider without ``api_keys`` is
    kept with an empty key list (acquire will raise ``NoLiveKeyError``); a
    provider without ``api_base`` is dropped.
    """
    if not isinstance(data, dict):
        raise ValueError('runtime.json must be a JSON object')
    cooldown = data.get('cooldown_seconds')
    if not isinstance(cooldown, (int, float)) or cooldown < 0:
        cooldown = DEFAULT_COOLDOWN_SECONDS
    raw_providers = data.get('providers') or {}
    if not isinstance(raw_providers, dict):
        raise ValueError("runtime.json 'providers' must be an object")

    providers = {}
    for name, entry in raw_providers.items():
        if not isinstance(entry, dict):
            continue
        api_base = entry.get('api_base')
        if not isinstance(api_base, str) or not api_base.strip():
            continue
        api_keys = entry.get('api_keys') or []
        if not isinstance(api_keys, list):
            continue
        keys = [
            _KeyState(api_key=str(k).strip())
            for k in api_keys
            if isinstance(k, str) and k.strip()
        ]
        prefixes = entry.get('model_prefixes') or []
        if not isinstance(prefixes, list):
            prefixes = []
        providers[str(name)] = _ProviderState(
            api_base=api_base.strip(),
            keys=keys,
            model_prefixes=tuple(str(p) for p in prefixes if isinstance(p, str)),
        )

    default = data.get('default_provider')
    if not isinstance(default, str) or default not in providers:
        default = next(iter(providers), '')

    return {
        'default_provider': default,
        'cooldown_seconds': float(cooldown),
        'providers': providers,
    }


def _maybe_reload():
    """Re-read runtime.json if its mtime changed since last load.

    Called at the top of every public entry point. Cheap when nothing changed
    (one ``stat`` call) — same trick ``runtime_config.get_active_model`` uses.
    """
    with _FILE_LOCK:
        try:
            mtime = RUNTIME_PATH.stat().st_mtime
        except FileNotFoundError:
            mtime = 0.0
            file_present = False
        else:
            file_present = True

        if mtime == _STATE['mtime']:
            return

        if not file_present:
            new = _bootstrap_from_env()
        else:
            try:
                with RUNTIME_PATH.open('r', encoding='utf-8') as f:
                    raw = json.load(f)
                new = _parse_config(raw)
                if not new['providers']:
                    # file present but contains no provider config → keep env
                    # fallback alive (e.g. only ``active_model`` written so far)
                    new = _bootstrap_from_env()
            except (OSError, ValueError):
                new = _bootstrap_from_env()

        _STATE['mtime'] = mtime
        _STATE['default_provider'] = new['default_provider']
        _STATE['cooldown_seconds'] = new['cooldown_seconds']
        _STATE['providers'] = new['providers']


# ──────────────────────────── Public API ──────────────────────────── #

def get_default_provider():
    _maybe_reload()
    return _STATE['default_provider']


def list_providers():
    """Return provider names in load order. For diagnostics / API surface."""
    _maybe_reload()
    return list(_STATE['providers'].keys())


def resolve_provider(model_name):
    """Map a model name to a provider via prefix match; fall back to default.

    A model that doesn't match any configured prefix still works — it goes
    through ``default_provider``. Callers shouldn't have to update config to
    try a brand-new model from an existing provider.
    """
    _maybe_reload()
    if isinstance(model_name, str) and model_name:
        for name, state in _STATE['providers'].items():
            for prefix in state.model_prefixes:
                if prefix and model_name.startswith(prefix):
                    return name
    return _STATE['default_provider']


def acquire(provider):
    """Pick the next live key for ``provider`` (round-robin).

    A key is "live" when it isn't permanently evicted and isn't currently in
    cooldown. Walks at most ``len(keys)`` slots so an all-dead pool fails fast
    instead of spinning. Returns a ``CredBundle`` the caller passes to
    ``LLMClient`` (and stores the ``key_id`` to feed back to ``report_failure``
    on error).
    """
    _maybe_reload()
    state = _STATE['providers'].get(provider)
    if state is None:
        raise UnknownProviderError(provider)
    n = len(state.keys)
    if n == 0:
        raise NoLiveKeyError(provider)
    now = time.time()
    with state.lock:
        for _ in range(n):
            idx = state.cursor % n
            state.cursor = (state.cursor + 1) % n
            ks = state.keys[idx]
            if ks.evicted:
                continue
            if ks.cooldown_until and ks.cooldown_until > now:
                continue
            # passed cooldown — clear the marker so it doesn't keep skipping
            ks.cooldown_until = 0.0
            return CredBundle(
                provider=provider,
                key_id=idx,
                api_key=ks.api_key,
                api_base=state.api_base,
            )
    raise NoLiveKeyError(provider)


def report_failure(provider, key_id, status_code):
    """Update the pool based on the upstream HTTP status.

    Rules:
      * 401/403 → permanent eviction (key is bad credentials, won't recover)
      * 429     → cooldown for ``cooldown_seconds`` (rate-limited, will recover)
      * else    → no-op (5xx / network blips shouldn't drain good keys)

    Best-effort: unknown provider, out-of-range ``key_id``, or non-int status
    are all swallowed so a buggy caller can't crash the pool.
    """
    _maybe_reload()
    state = _STATE['providers'].get(provider)
    if state is None:
        return
    if not isinstance(key_id, int) or key_id < 0 or key_id >= len(state.keys):
        return
    code = None
    try:
        code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return
    if code is None:
        return
    ks = state.keys[key_id]
    with state.lock:
        if code in EVICT_STATUS_CODES:
            ks.evicted = True
        elif code in COOLDOWN_STATUS_CODES:
            ks.cooldown_until = time.time() + _STATE['cooldown_seconds']


def snapshot():
    """Read-only view of the current pool. For debug/observability tools.

    Not intended for hot paths — copies the full state under the file lock.
    """
    _maybe_reload()
    out = {}
    for name, state in _STATE['providers'].items():
        out[name] = {
            'api_base': state.api_base,
            'model_prefixes': list(state.model_prefixes),
            'keys': [
                {
                    'key_id': i,
                    'evicted': k.evicted,
                    'cooldown_until': k.cooldown_until,
                    # never expose raw key — only a fingerprint
                    'api_key_tail': (k.api_key[-4:] if len(k.api_key) >= 4 else ''),
                }
                for i, k in enumerate(state.keys)
            ],
        }
    return {
        'default_provider': _STATE['default_provider'],
        'cooldown_seconds': _STATE['cooldown_seconds'],
        'providers': out,
    }


def reset_cache():
    """Force the next call to re-read runtime.json. Tests use this."""
    with _FILE_LOCK:
        _STATE['mtime'] = -1.0
        _STATE['default_provider'] = ''
        _STATE['providers'] = {}


def _snapshot_full_keys():
    """In-process-only reverse map ``api_key -> (provider, key_id)``.

    Internal API for the agents-SDK httpx response hook, which only sees the
    raw ``Authorization`` bearer and needs to feed ``report_failure`` the
    right ``(provider, key_id)`` pair. Never exposed over the wire — keys are
    secrets — and intentionally distinct from :func:`snapshot` (which redacts
    to ``api_key_tail``).
    """
    _maybe_reload()
    out = {}
    for name, state in _STATE['providers'].items():
        for i, ks in enumerate(state.keys):
            if ks.api_key:
                out[ks.api_key] = (name, i)
    return out
