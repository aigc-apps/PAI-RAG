import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
from unittest.mock import MagicMock, patch
from extensions.trace.baggage_processor import LoongSuiteBaggageSpanProcessor
from extensions.trace.context import (
    init_custom_context,
    get_context_vars,
    set_context_var,
    set_request_id,
    get_request_id,
    _request_id_var,
    custom_context_vars,
)


class TestLoongSuiteBaggageSpanProcessor:
    def test_default_prefixes(self):
        processor = LoongSuiteBaggageSpanProcessor()
        assert not processor._allow_all

    def test_allow_all_when_no_prefixes(self):
        processor = LoongSuiteBaggageSpanProcessor(allowed_prefixes=set())
        assert processor._allow_all is True

    def test_should_process_matching_key(self):
        processor = LoongSuiteBaggageSpanProcessor(allowed_prefixes={"traffic.llm_sdk."})
        assert processor._should_process_key("traffic.llm_sdk.user_id") is True

    def test_should_not_process_non_matching_key(self):
        processor = LoongSuiteBaggageSpanProcessor(allowed_prefixes={"traffic.llm_sdk."})
        assert processor._should_process_key("other.key") is False

    def test_should_process_all_when_allow_all(self):
        processor = LoongSuiteBaggageSpanProcessor(allowed_prefixes=set())
        assert processor._should_process_key("any.key") is True

    def test_strip_matching_prefix(self):
        processor = LoongSuiteBaggageSpanProcessor(strip_prefixes={"traffic.llm_sdk."})
        assert processor._strip_prefix("traffic.llm_sdk.user_id") == "user_id"

    def test_strip_no_match(self):
        processor = LoongSuiteBaggageSpanProcessor(strip_prefixes={"traffic.llm_sdk."})
        assert processor._strip_prefix("other.key") == "other.key"

    @patch("extensions.trace.baggage_processor.get_all_baggage")
    def test_on_start_sets_attributes(self, mock_get_baggage):
        mock_get_baggage.return_value = {
            "traffic.llm_sdk.user_id": "123",
            "other.key": "ignored",
        }
        processor = LoongSuiteBaggageSpanProcessor(
            allowed_prefixes={"traffic.llm_sdk."},
            strip_prefixes={"traffic.llm_sdk."},
        )
        mock_span = MagicMock()
        processor.on_start(mock_span, parent_context=None)
        mock_span.set_attribute.assert_called_once_with("user_id", "123")


class TestTraceContext:
    def setup_method(self):
        _request_id_var.set(None)
        custom_context_vars.clear()

    def test_set_and_get_request_id(self):
        set_request_id("req-123")
        assert get_request_id() == "req-123"

    def test_default_request_id_is_none(self):
        assert get_request_id() is None

    def test_init_custom_context(self):
        init_custom_context(["key1", "key2"])
        assert "key1" in custom_context_vars
        assert "key2" in custom_context_vars

    def test_set_and_get_context_var(self):
        init_custom_context(["my_key"])
        set_context_var("my_key", "my_value")
        vars = get_context_vars()
        assert vars["my_key"] == "my_value"

    def test_set_unknown_key_warns(self):
        # Should not raise, just log a warning
        set_context_var("unknown_key", "value")

    def test_get_context_vars_empty(self):
        result = get_context_vars()
        assert result == {}
