"""Per-request LLM client wiring for the Agents-SDK code path.

Replaces the older "install a process-global ``AsyncOpenAI`` as the SDK
default" approach (which silently degenerated into last-write-wins under
concurrency — every in-flight request shared whichever key was installed
most recently). The new flow:

  * ``ensure_sdk_runtime()`` is a one-shot init that flips the SDK's default
    API to ``chat_completions`` and lazily builds a process-wide
    ``httpx.AsyncClient`` whose response event-hook reports 401/403/429 back
    to :mod:`provider_pool`. Safe to call on every request handler.
  * ``acquire_request_model(model_name)`` is called **per request** by the
    runner. It picks a fresh ``(api_key, api_base)`` from the pool and
    returns an ``OpenAIChatCompletionsModel`` bound to a per-request
    ``AsyncOpenAI`` that shares the global httpx pool (so we keep keep-alive
    + connection reuse). The caller passes the returned Model into
    :func:`backend.agents_sdk.agent_factory.build`, and from there it goes
    into the ``Agent`` instance for exactly that run.

The httpx response hook reverse-maps the request's ``Authorization`` header
back to a ``(provider, key_id)`` via :func:`provider_pool._snapshot_full_keys`,
so eviction/cooldown actually fires for the specific key that returned the
failure — not for whichever key happens to be installed globally at the
moment the error bubbles up.
"""
from __future__ import annotations

import logging
import threading
from typing import Tuple

import httpx
from agents import AsyncOpenAI, set_default_openai_api
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

import provider_pool
import runtime_config
import settings as config

logger = logging.getLogger(__name__)

_INIT_LOCK = threading.Lock()
_SHARED_HTTP_CLIENT: httpx.AsyncClient | None = None
_API_SET = False


async def _on_response(response: httpx.Response) -> None:
    """httpx response hook: route 401/403/429 to ``provider_pool.report_failure``.

    Must be ``async`` — ``httpx.AsyncClient`` awaits hooks unconditionally
    (a sync function returning ``None`` triggers ``TypeError: object NoneType
    can't be used in 'await' expression``). Identifies the owning key by the
    request's ``Authorization`` header. Hook fires after the response head
    is read, before the body is consumed — we do not touch the body. Any
    unrecognised bearer (request that bypassed the pool, or a key that has
    since been removed) is silently ignored.
    """
    code = response.status_code
    if code not in (401, 403, 429):
        return
    auth = response.request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return
    key = auth[len('Bearer '):].strip()
    if not key:
        return
    mapping = provider_pool._snapshot_full_keys()
    entry = mapping.get(key)
    if entry is None:
        return
    provider, key_id = entry
    provider_pool.report_failure(provider, key_id, code)


def _ensure_shared_http_client() -> httpx.AsyncClient:
    """Lazy-init the process-wide httpx client. Idempotent + thread-safe."""
    global _SHARED_HTTP_CLIENT, _API_SET
    if _SHARED_HTTP_CLIENT is not None:
        return _SHARED_HTTP_CLIENT
    with _INIT_LOCK:
        if _SHARED_HTTP_CLIENT is None:
            # Timeouts: connect/write/pool kept tight; read is long because
            # chat-completions streaming holds the connection open for the
            # entire generation. Limits sized for ``--workers 4`` with a few
            # hundred concurrent agent runs sharing the pool.
            _SHARED_HTTP_CLIENT = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0),
                limits=httpx.Limits(max_connections=200, max_keepalive_connections=100),
                event_hooks={'response': [_on_response]},
            )
        if not _API_SET:
            set_default_openai_api('chat_completions')
            _API_SET = True
    return _SHARED_HTTP_CLIENT


def ensure_sdk_runtime() -> None:
    """Idempotent one-shot init for SDK defaults + shared HTTP client.

    Old callers (``server.py:_sdk_response_stream`` / ``_sdk_chat_stream``)
    still call this; it now just primes the shared client and flips the
    default API on first call. Per-request key selection lives in
    :func:`acquire_request_model` and is invoked by the runner.
    """
    _ensure_shared_http_client()


def _resolve_credentials() -> Tuple[str, str, int | None]:
    """Pick the next live key for the currently active model.

    Falls back to the env ``API_KEY`` (key_id=None) so a misconfigured pool
    doesn't take the SDK offline; matches the legacy ``make_llm_client``
    behavior to keep deployments without ``runtime.json`` working as before.
    """
    model = runtime_config.get_active_model()
    provider = provider_pool.resolve_provider(model)
    try:
        bundle = provider_pool.acquire(provider)
    except (provider_pool.NoLiveKeyError, provider_pool.UnknownProviderError):
        bundle = None
    if bundle is not None:
        return bundle.api_key, bundle.api_base, bundle.key_id
    api_key = getattr(config, 'API_KEY', '') or ''
    if not api_key:
        # Re-raise the original pool error rather than handing the SDK an
        # empty bearer — the failure mode of an empty key is opaque (401 on
        # the first call) versus an explicit NoLiveKey here.
        raise provider_pool.NoLiveKeyError(provider)
    api_base = getattr(config, 'API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    return api_key, api_base, None


def acquire_request_model(model_name: str) -> OpenAIChatCompletionsModel:
    """Build a per-request ``Model`` instance bound to a freshly-acquired key.

    Called by :mod:`backend.agents_sdk.runner` once per agent run. The
    returned ``OpenAIChatCompletionsModel`` carries its own ``AsyncOpenAI``
    whose ``api_key`` is the round-robin pick from the pool, but the
    underlying httpx connection pool is shared process-wide via
    :func:`_ensure_shared_http_client` — so we keep TCP/TLS keep-alive while
    still rotating credentials per request.

    The Authorization header on every outbound chat-completions request is
    therefore the per-request bearer, and the shared response hook can
    reverse-map it back to ``(provider, key_id)`` for failure feedback.
    """
    http_client = _ensure_shared_http_client()
    api_key, api_base, key_id = _resolve_credentials()
    # Log key tail per acquisition so multi-key round-robin distribution is
    # observable in worker logs without exposing the full secret. Watch for
    # the tail rotating across concurrent requests.
    logger.info(
        'acquired llm key tail=%s key_id=%s model=%s',
        api_key[-4:] if len(api_key) >= 4 else '',
        key_id,
        model_name,
    )
    openai_client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base,
        http_client=http_client,
    )
    return OpenAIChatCompletionsModel(model=model_name, openai_client=openai_client)
