"""Per-request key distribution for the agents-SDK code path.

Covers the wiring between ``provider_pool`` (round-robin) and
``backend.agents_sdk.runtime_setup.acquire_request_model`` (the per-request
``Model`` factory). The pool itself is exercised in ``test_provider_pool.py``;
here we verify the integration points the runner depends on:

  1. Concurrent acquisitions distribute across keys (no last-write-wins).
  2. A 4xx/429 from the shared httpx client routes ``report_failure`` to the
     specific key that owned the failing request.
  3. An empty pool with no env fallback still raises ``NoLiveKeyError``.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import provider_pool


def _write_config(path: Path, providers: dict, default: str = 'qwen', cooldown: int = 300) -> None:
    old_mtime = path.stat().st_mtime if path.exists() else 0.0
    path.write_text(json.dumps({
        'default_provider': default,
        'cooldown_seconds': cooldown,
        'providers': providers,
    }), encoding='utf-8')
    new_time = max(old_mtime, path.stat().st_mtime) + 5
    os.utime(path, (new_time, new_time))


@pytest.fixture(autouse=True)
def _isolated_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(provider_pool, 'RUNTIME_DIR', tmp_path)
    monkeypatch.setattr(provider_pool, 'RUNTIME_PATH', tmp_path / 'runtime.json')
    provider_pool.reset_cache()
    yield
    provider_pool.reset_cache()


# ───── 1. round-robin under concurrency ───── #

def test_acquire_request_model_round_robins_under_concurrency(tmp_path, monkeypatch):
    """8 concurrent acquires across 2 keys must split 4/4 (round-robin is atomic)."""
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'https://q.test/v1', 'api_keys': ['k0', 'k1'], 'model_prefixes': ['qwen-']},
    })

    from backend.agents_sdk import runtime_setup
    seen = []

    class _FakeModel:
        def __init__(self, model, openai_client, **_kwargs):
            self.model = model
            self.openai_client = openai_client
            seen.append(openai_client.api_key)

    monkeypatch.setattr(runtime_setup, 'OpenAIChatCompletionsModel', _FakeModel)

    async def _go():
        return await asyncio.gather(*[
            asyncio.to_thread(runtime_setup.acquire_request_model, 'qwen-plus')
            for _ in range(8)
        ])

    asyncio.run(_go())
    assert Counter(seen) == Counter({'k0': 4, 'k1': 4}), seen


# ───── 2. failure feedback routes to the right key ───── #

def test_response_hook_reports_429_to_owning_key(tmp_path, monkeypatch):
    """A synthesised 429 response with Authorization: Bearer k1 must call
    report_failure('qwen', 1, 429) — and nothing else."""
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'https://q.test/v1', 'api_keys': ['k0', 'k1']},
    })

    from backend.agents_sdk import runtime_setup

    calls: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        provider_pool, 'report_failure',
        lambda provider, key_id, code: calls.append((provider, key_id, code)),
    )

    request = httpx.Request(
        'POST', 'https://q.test/v1/chat/completions',
        headers={'Authorization': 'Bearer k1'},
    )
    response = httpx.Response(status_code=429, request=request)
    asyncio.run(runtime_setup._on_response(response))

    assert calls == [('qwen', 1, 429)]


def test_response_hook_ignores_2xx(tmp_path, monkeypatch):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'https://q.test/v1', 'api_keys': ['k0']},
    })
    from backend.agents_sdk import runtime_setup

    calls: list = []
    monkeypatch.setattr(
        provider_pool, 'report_failure',
        lambda provider, key_id, code: calls.append((provider, key_id, code)),
    )
    request = httpx.Request(
        'POST', 'https://q.test/v1/chat/completions',
        headers={'Authorization': 'Bearer k0'},
    )
    asyncio.run(runtime_setup._on_response(httpx.Response(status_code=200, request=request)))
    asyncio.run(runtime_setup._on_response(httpx.Response(status_code=500, request=request)))
    assert calls == []


def test_response_hook_no_op_when_authorization_missing(tmp_path, monkeypatch):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'https://q.test/v1', 'api_keys': ['k0']},
    })
    from backend.agents_sdk import runtime_setup

    calls: list = []
    monkeypatch.setattr(
        provider_pool, 'report_failure',
        lambda provider, key_id, code: calls.append((provider, key_id, code)),
    )
    request = httpx.Request('POST', 'https://q.test/v1/chat/completions')
    asyncio.run(runtime_setup._on_response(httpx.Response(status_code=401, request=request)))
    assert calls == []


def test_response_hook_unknown_bearer_is_no_op(tmp_path, monkeypatch):
    """Authorization for a key not in the pool (e.g. stale) shouldn't crash
    or fabricate a key_id."""
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'https://q.test/v1', 'api_keys': ['k0']},
    })
    from backend.agents_sdk import runtime_setup

    calls: list = []
    monkeypatch.setattr(
        provider_pool, 'report_failure',
        lambda provider, key_id, code: calls.append((provider, key_id, code)),
    )
    request = httpx.Request(
        'POST', 'https://q.test/v1/chat/completions',
        headers={'Authorization': 'Bearer this-key-was-never-in-the-pool'},
    )
    asyncio.run(runtime_setup._on_response(httpx.Response(status_code=401, request=request)))
    assert calls == []


# ───── 3. empty pool + no env fallback ───── #

def test_acquire_raises_when_pool_empty_and_no_env_key(tmp_path, monkeypatch):
    monkeypatch.setattr(provider_pool.config, 'API_KEY', '', raising=False)
    from backend.agents_sdk import runtime_setup
    with pytest.raises(provider_pool.NoLiveKeyError):
        runtime_setup.acquire_request_model('qwen-plus')


# ───── 4. internal reverse-lookup helper ───── #

def test_snapshot_full_keys_returns_provider_and_keyid(tmp_path):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'https://q.test/v1', 'api_keys': ['kA', 'kB']},
        'deepseek': {'api_base': 'https://d.test/v1', 'api_keys': ['kZ']},
    })
    mapping = provider_pool._snapshot_full_keys()
    assert mapping == {
        'kA': ('qwen', 0),
        'kB': ('qwen', 1),
        'kZ': ('deepseek', 0),
    }
