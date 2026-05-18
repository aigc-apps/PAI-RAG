"""Idempotent setup that points the OpenAI Agents SDK at our LLM endpoint.

PAI-RAG talks to a Chat-Completions-compatible endpoint (DashScope/Qwen by
default). The SDK defaults to the Responses API on api.openai.com, so on
first use we (a) flip the default API to ``chat_completions`` and
(b) install an ``AsyncOpenAI`` client whose ``api_key`` / ``base_url`` come
from the same provider-pool path as :func:`make_llm_client`.

The runtime model can change mid-process via ``/v1/models/active``. The
provider pool keys off ``runtime_config.get_active_model()``, so we re-read
on each ``ensure_sdk_runtime`` call and refresh the client when the
resolved provider (api_key, api_base) changed since last setup.
"""
from __future__ import annotations

import threading
from typing import Tuple

from agents import (
    AsyncOpenAI,
    set_default_openai_api,
    set_default_openai_client,
)

import provider_pool
import runtime_config
import settings as config

_LOCK = threading.Lock()
_CONFIGURED: Tuple[str, str] | None = None  # (api_key_id, api_base)
_API_SET = False


def _resolve_credentials() -> Tuple[str, str, str]:
    """Return ``(api_key, api_base, key_id)`` matching :func:`make_llm_client`.

    Falls back to the env-configured key when the provider pool is empty so
    a misconfigured pool doesn't take the SDK runtime offline.
    """
    model = runtime_config.get_active_model()
    provider = provider_pool.resolve_provider(model)
    bundle = None
    try:
        bundle = provider_pool.acquire(provider)
    except (provider_pool.NoLiveKeyError, provider_pool.UnknownProviderError):
        bundle = None
    if bundle is not None:
        return bundle.api_key, bundle.api_base, bundle.key_id or ''
    api_key = getattr(config, 'API_KEY', '') or ''
    api_base = getattr(config, 'API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    return api_key, api_base, ''


def ensure_sdk_runtime() -> None:
    """Configure the SDK's default OpenAI client + API. Safe to call from
    every request handler; only re-installs the client if credentials moved.
    """
    global _CONFIGURED, _API_SET
    api_key, api_base, key_id = _resolve_credentials()
    fingerprint = (key_id or api_key, api_base)
    with _LOCK:
        if not _API_SET:
            set_default_openai_api('chat_completions')
            _API_SET = True
        if _CONFIGURED == fingerprint:
            return
        client = AsyncOpenAI(api_key=api_key, base_url=api_base)
        set_default_openai_client(client, use_for_tracing=False)
        _CONFIGURED = fingerprint
