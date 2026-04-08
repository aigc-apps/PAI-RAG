import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from common.i18n import I18n, set_request_locale, get_request_locale, _locale_context


class TestI18n:
    def setup_method(self):
        """Reset locale context before each test."""
        _locale_context.set(None)

    def test_singleton(self):
        a = I18n()
        b = I18n()
        assert a is b

    def test_default_locale_is_zh(self):
        i18n = I18n()
        assert i18n._get_locale() == "zh"

    def test_translate_zh_key(self):
        i18n = I18n()
        _locale_context.set("zh")
        result = i18n.t("api.success.create")
        assert result == "创建成功"

    def test_translate_en_key(self):
        i18n = I18n()
        _locale_context.set("en")
        result = i18n.t("api.success.create")
        assert result == "Created successfully"

    def test_translate_with_params(self):
        i18n = I18n()
        _locale_context.set("en")
        result = i18n.t("api.error.not_found", resource="KB", id="123")
        assert "KB" in result
        assert "123" in result

    def test_translate_with_dollar_params(self):
        i18n = I18n()
        _locale_context.set("zh")
        result = i18n.t("api.error.not_found", resource="知识库", id="kb1")
        assert "知识库" in result
        assert "kb1" in result

    def test_missing_key_returns_key(self):
        i18n = I18n()
        result = i18n.t("nonexistent.key.path")
        assert result == "nonexistent.key.path"

    def test_non_string_value_returns_key(self):
        i18n = I18n()
        # "api.success" is a dict, not a string
        result = i18n.t("api.success")
        assert result == "api.success"

    def test_set_locale(self):
        i18n = I18n()
        i18n.set_locale("en")
        _locale_context.set(None)  # no request locale
        assert i18n._get_locale() == "en"
        # Reset
        i18n.set_locale("zh")

    def test_set_unsupported_locale_ignored(self):
        i18n = I18n()
        original = i18n._default_locale
        i18n.set_locale("fr")
        assert i18n._default_locale == original


class TestSetRequestLocale:
    def setup_method(self):
        _locale_context.set(None)

    def test_set_valid_locale(self):
        set_request_locale("en")
        assert get_request_locale() == "en"

    def test_set_invalid_locale_defaults_to_zh(self):
        set_request_locale("fr")
        assert get_request_locale() == "zh"


class TestGetRequestLocale:
    def setup_method(self):
        _locale_context.set(None)

    def test_default_is_zh(self):
        assert get_request_locale() == "zh"

    def test_returns_set_locale(self):
        _locale_context.set("en")
        assert get_request_locale() == "en"
