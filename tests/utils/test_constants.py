import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from utils.constants import try_get_int_env


class TestTryGetIntEnv:
    def test_valid_int_env(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "42")
        assert try_get_int_env("TEST_INT_VAR") == 42

    def test_invalid_string_env(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "not_a_number")
        assert try_get_int_env("TEST_INT_VAR") is None

    def test_unset_with_default(self, monkeypatch):
        monkeypatch.delenv("TEST_UNSET_VAR", raising=False)
        assert try_get_int_env("TEST_UNSET_VAR", 99) == 99

    def test_unset_without_default(self, monkeypatch):
        monkeypatch.delenv("TEST_UNSET_VAR", raising=False)
        assert try_get_int_env("TEST_UNSET_VAR") is None

    def test_float_string_returns_none(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "3.14")
        assert try_get_int_env("TEST_INT_VAR") is None
