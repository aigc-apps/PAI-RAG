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
    monkeypatch.setattr(runtime_config, 'ACTIVE_MODEL_PATH', tmp_path / 'active_model.json')
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
    runtime_config.ACTIVE_MODEL_PATH.write_text(json.dumps({'model': 'second'}), encoding='utf-8')
    new_time = runtime_config.ACTIVE_MODEL_PATH.stat().st_mtime + 5
    os.utime(runtime_config.ACTIVE_MODEL_PATH, (new_time, new_time))
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
    runtime_config.ACTIVE_MODEL_PATH.write_text('not json{{', encoding='utf-8')
    assert runtime_config.get_active_model() == 'fallback-model'


def test_empty_string_in_file_falls_back(monkeypatch):
    monkeypatch.setattr(runtime_config.config, 'MODEL', 'fallback-model', raising=False)
    runtime_config.ACTIVE_MODEL_PATH.write_text(json.dumps({'model': '   '}), encoding='utf-8')
    assert runtime_config.get_active_model() == 'fallback-model'


def test_atomic_write_no_tempfile_on_success(tmp_path):
    runtime_config.set_active_model('clean-model')
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith('active_model.') and p.name != 'active_model.json']
    assert leftovers == []


def test_failed_replace_cleans_up_tempfile(tmp_path, monkeypatch):
    real_replace = os.replace

    def boom(*args, **kwargs):
        raise OSError('simulated failure')

    monkeypatch.setattr(runtime_config.os, 'replace', boom)
    with pytest.raises(OSError):
        runtime_config.set_active_model('will-fail')
    monkeypatch.setattr(runtime_config.os, 'replace', real_replace)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith('active_model.') and p.name != 'active_model.json']
    assert leftovers == []
