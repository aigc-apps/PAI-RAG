"""Unit tests for provider_pool — round-robin, failure rules, hot reload."""
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import provider_pool


@pytest.fixture(autouse=True)
def _isolated_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(provider_pool, 'RUNTIME_DIR', tmp_path)
    monkeypatch.setattr(provider_pool, 'RUNTIME_PATH', tmp_path / 'runtime.json')
    provider_pool.reset_cache()
    yield
    provider_pool.reset_cache()


def _write_config(path, providers, default='qwen', cooldown=300):
    # On filesystems with second-resolution mtime, two writes within the same
    # second otherwise share an mtime and the cache wouldn't reload. Bump
    # relative to whatever was there before, not just the fresh write's time.
    old_mtime = path.stat().st_mtime if path.exists() else 0.0
    path.write_text(json.dumps({
        'default_provider': default,
        'cooldown_seconds': cooldown,
        'providers': providers,
    }), encoding='utf-8')
    new_time = max(old_mtime, path.stat().st_mtime) + 5
    os.utime(path, (new_time, new_time))


# ───── env fallback ───── #

def test_missing_file_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(provider_pool.config, 'API_KEY', 'sk-env', raising=False)
    monkeypatch.setattr(provider_pool.config, 'API_BASE', 'https://example.test/v1', raising=False)
    bundle = provider_pool.acquire('qwen')
    assert bundle.provider == 'qwen'
    assert bundle.api_key == 'sk-env'
    assert bundle.api_base == 'https://example.test/v1'
    assert bundle.key_id == 0


def test_missing_file_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(provider_pool.config, 'API_KEY', 'sk-env', raising=False)
    with pytest.raises(provider_pool.UnknownProviderError):
        provider_pool.acquire('deepseek')


def test_missing_file_no_env_key_raises(monkeypatch):
    monkeypatch.setattr(provider_pool.config, 'API_KEY', '', raising=False)
    with pytest.raises(provider_pool.NoLiveKeyError):
        provider_pool.acquire('qwen')


# ───── round-robin ───── #

def test_round_robin_three_keys(tmp_path):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {
            'api_base': 'https://q.test/v1',
            'api_keys': ['k0', 'k1', 'k2'],
            'model_prefixes': ['qwen-'],
        },
    })
    seen = [provider_pool.acquire('qwen').key_id for _ in range(6)]
    assert seen == [0, 1, 2, 0, 1, 2]


def test_single_key_loops(tmp_path):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'https://q.test/v1', 'api_keys': ['only']},
    })
    bundles = [provider_pool.acquire('qwen') for _ in range(4)]
    assert all(b.key_id == 0 and b.api_key == 'only' for b in bundles)


# ───── failure semantics ───── #

def test_401_evicts_permanently(tmp_path):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'https://q.test/v1', 'api_keys': ['good', 'bad']},
    })
    # Grab them once each so cursor is past idx 1
    a = provider_pool.acquire('qwen')
    b = provider_pool.acquire('qwen')
    assert {a.key_id, b.key_id} == {0, 1}
    # Mark 'bad' (idx 1) as evicted via 401
    provider_pool.report_failure('qwen', 1, 401)
    seen = {provider_pool.acquire('qwen').key_id for _ in range(10)}
    assert seen == {0}


def test_403_also_evicts(tmp_path):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'https://q.test/v1', 'api_keys': ['k0', 'k1']},
    })
    provider_pool.report_failure('qwen', 0, 403)
    seen = {provider_pool.acquire('qwen').key_id for _ in range(8)}
    assert seen == {1}


def test_429_cools_down_then_revives(tmp_path, monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr(provider_pool.time, 'time', lambda: fake_now[0])
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'https://q.test/v1', 'api_keys': ['k0', 'k1']},
    }, cooldown=300)
    provider_pool.report_failure('qwen', 0, 429)
    fake_now[0] = 1100.0  # 100s in, still cooling
    seen = {provider_pool.acquire('qwen').key_id for _ in range(6)}
    assert seen == {1}
    fake_now[0] = 1400.0  # past 300s cooldown
    seen2 = {provider_pool.acquire('qwen').key_id for _ in range(6)}
    assert seen2 == {0, 1}


def test_500_does_not_evict(tmp_path):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'https://q.test/v1', 'api_keys': ['k0']},
    })
    provider_pool.report_failure('qwen', 0, 500)
    provider_pool.report_failure('qwen', 0, 503)
    bundle = provider_pool.acquire('qwen')
    assert bundle.key_id == 0


def test_all_dead_raises(tmp_path):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'https://q.test/v1', 'api_keys': ['k0', 'k1']},
    })
    provider_pool.report_failure('qwen', 0, 401)
    provider_pool.report_failure('qwen', 1, 401)
    with pytest.raises(provider_pool.NoLiveKeyError):
        provider_pool.acquire('qwen')


def test_report_failure_tolerates_garbage_inputs(tmp_path):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'https://q.test/v1', 'api_keys': ['k0']},
    })
    provider_pool.report_failure('nope', 0, 401)        # unknown provider
    provider_pool.report_failure('qwen', 99, 401)       # out of range
    provider_pool.report_failure('qwen', 0, None)       # None status
    provider_pool.report_failure('qwen', 0, 'oops')     # bad type
    bundle = provider_pool.acquire('qwen')
    assert bundle.key_id == 0


# ───── provider routing ───── #

def test_resolve_by_prefix(tmp_path):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'q', 'api_keys': ['k'], 'model_prefixes': ['qwen-']},
        'deepseek': {'api_base': 'd', 'api_keys': ['k'], 'model_prefixes': ['deepseek-']},
        'zhipu': {'api_base': 'z', 'api_keys': ['k'], 'model_prefixes': ['glm-']},
    }, default='qwen')
    assert provider_pool.resolve_provider('qwen-plus') == 'qwen'
    assert provider_pool.resolve_provider('deepseek-chat') == 'deepseek'
    assert provider_pool.resolve_provider('glm-4-plus') == 'zhipu'


def test_unknown_model_falls_back_to_default(tmp_path):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'q', 'api_keys': ['k'], 'model_prefixes': ['qwen-']},
        'deepseek': {'api_base': 'd', 'api_keys': ['k'], 'model_prefixes': ['deepseek-']},
    }, default='qwen')
    assert provider_pool.resolve_provider('mystery-model') == 'qwen'


def test_resolve_with_empty_or_none_returns_default(tmp_path):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'q', 'api_keys': ['k']},
    }, default='qwen')
    assert provider_pool.resolve_provider('') == 'qwen'
    assert provider_pool.resolve_provider(None) == 'qwen'


# ───── hot reload ───── #

def test_mtime_change_triggers_reload(tmp_path):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'q', 'api_keys': ['old']},
    })
    assert provider_pool.acquire('qwen').api_key == 'old'
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'q', 'api_keys': ['new']},
    })
    assert provider_pool.acquire('qwen').api_key == 'new'


def test_corrupt_file_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setattr(provider_pool.config, 'API_KEY', 'sk-env', raising=False)
    monkeypatch.setattr(provider_pool.config, 'API_BASE', 'https://example.test/v1', raising=False)
    provider_pool.RUNTIME_PATH.write_text('not json{{', encoding='utf-8')
    bundle = provider_pool.acquire('qwen')
    assert bundle.api_key == 'sk-env'


# ───── concurrency ───── #

def test_concurrent_acquire_distributes_evenly(tmp_path):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'q', 'api_keys': ['k0', 'k1']},
    })
    counter = Counter()
    lock = threading.Lock()

    def grab():
        for _ in range(50):
            b = provider_pool.acquire('qwen')
            with lock:
                counter[b.key_id] += 1

    threads = [threading.Thread(target=grab) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 4 threads × 50 = 200 acquires; with round-robin under a lock, each key
    # should get exactly 100 (deterministic) — generous tolerance is fine.
    assert sum(counter.values()) == 200
    assert abs(counter[0] - counter[1]) <= 4


# ───── snapshot ───── #

def test_snapshot_does_not_leak_full_keys(tmp_path):
    _write_config(provider_pool.RUNTIME_PATH, {
        'qwen': {'api_base': 'q', 'api_keys': ['sk-supersecret-1234']},
    })
    snap = provider_pool.snapshot()
    qwen = snap['providers']['qwen']
    assert qwen['keys'][0]['api_key_tail'] == '1234'
    flat = json.dumps(snap)
    assert 'sk-supersecret-1234' not in flat
