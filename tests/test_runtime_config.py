"""Unit tests for runtime_config.get_active_model / set_active_model."""
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import runtime_config


@pytest.fixture(autouse=True)
def _isolated_runtime_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_config, 'RUNTIME_DIR', tmp_path)
    monkeypatch.setattr(runtime_config, 'RUNTIME_PATH', tmp_path / 'runtime.json')
    runtime_config.reset_cache()
    yield
    runtime_config.reset_cache()


def test_missing_file_returns_config_fallback(monkeypatch):
    monkeypatch.setattr(runtime_config.config, 'MODEL', 'qwen-plus', raising=False)
    assert runtime_config.get_active_model() == 'qwen-plus'


def test_set_then_get_roundtrip():
    runtime_config.set_active_model('qwen-max')
    assert runtime_config.get_active_model() == 'qwen-max'


def test_overwrite():
    runtime_config.set_active_model('a-model')
    runtime_config.set_active_model('b-model')
    assert runtime_config.get_active_model() == 'b-model'


def test_external_write_picked_up_via_mtime(tmp_path):
    runtime_config.set_active_model('first')
    assert runtime_config.get_active_model() == 'first'
    runtime_config.RUNTIME_PATH.write_text(json.dumps({'active_model': 'second'}), encoding='utf-8')
    new_time = runtime_config.RUNTIME_PATH.stat().st_mtime + 5
    os.utime(runtime_config.RUNTIME_PATH, (new_time, new_time))
    assert runtime_config.get_active_model() == 'second'


def test_persistence_across_cache_reset(monkeypatch):
    runtime_config.set_active_model('persisted-model')
    runtime_config.reset_cache()
    assert runtime_config.get_active_model() == 'persisted-model'


@pytest.mark.parametrize('bad', ['', '   ', 'has space', 'tab\there', 'new\nline', 'x' * 201, None, 123, []])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(ValueError):
        runtime_config.set_active_model(bad)


def test_corrupt_file_falls_back_to_config(monkeypatch):
    monkeypatch.setattr(runtime_config.config, 'MODEL', 'fallback-model', raising=False)
    runtime_config.RUNTIME_PATH.write_text('not json{{', encoding='utf-8')
    assert runtime_config.get_active_model() == 'fallback-model'


def test_empty_string_in_file_falls_back(monkeypatch):
    monkeypatch.setattr(runtime_config.config, 'MODEL', 'fallback-model', raising=False)
    runtime_config.RUNTIME_PATH.write_text(json.dumps({'active_model': '   '}), encoding='utf-8')
    assert runtime_config.get_active_model() == 'fallback-model'


def test_atomic_write_no_tempfile_on_success(tmp_path):
    runtime_config.set_active_model('clean-model')
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith('runtime.') and p.name != 'runtime.json']
    assert leftovers == []


def test_failed_replace_cleans_up_tempfile(tmp_path, monkeypatch):
    real_replace = os.replace

    def boom(*args, **kwargs):
        raise OSError('simulated failure')

    monkeypatch.setattr(runtime_config.os, 'replace', boom)
    with pytest.raises(OSError):
        runtime_config.set_active_model('will-fail')
    monkeypatch.setattr(runtime_config.os, 'replace', real_replace)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith('runtime.') and p.name != 'runtime.json']
    assert leftovers == []


def test_set_active_model_preserves_provider_keys(tmp_path):
    """set_active_model must do read-modify-write so the multi-provider config
    (owned by provider_pool) is not clobbered when toggling the active model."""
    runtime_config.RUNTIME_PATH.write_text(json.dumps({
        'default_provider': 'qwen',
        'cooldown_seconds': 300,
        'providers': {'qwen': {'api_base': 'q', 'api_keys': ['k0']}},
    }), encoding='utf-8')
    runtime_config.set_active_model('qwen-max')
    data = json.loads(runtime_config.RUNTIME_PATH.read_text(encoding='utf-8'))
    assert data['active_model'] == 'qwen-max'
    assert data['default_provider'] == 'qwen'
    assert data['cooldown_seconds'] == 300
    assert data['providers'] == {'qwen': {'api_base': 'q', 'api_keys': ['k0']}}
