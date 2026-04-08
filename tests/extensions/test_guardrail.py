import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from extensions.guardrail.guardrail_check import GuardrailChecker, TextCheckResult


class TestTextCheckResult:
    def test_defaults(self):
        result = TextCheckResult()
        assert result.reject is False
        assert result.reason is None
        assert result.risk_level == "low"
        assert result.advice is None

    def test_with_values(self):
        result = TextCheckResult(reject=True, reason="harmful", risk_level="high", advice="safe content")
        assert result.reject is True
        assert result.reason == "harmful"


class TestGuardrailChecker:
    @patch("extensions.guardrail.guardrail_check.Client")
    @patch("extensions.guardrail.guardrail_check.Config")
    def _make_checker(self, mock_config, mock_client_cls):
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        checker = GuardrailChecker(
            access_key_id="ak",
            access_key_secret="sk",
            region_id="cn-hangzhou",
            endpoint="https://green.cn-hangzhou.aliyuncs.com",
        )
        return checker, mock_client

    @patch("extensions.guardrail.guardrail_check.Client")
    @patch("extensions.guardrail.guardrail_check.Config")
    async def test_empty_text_returns_safe(self, mock_config, mock_client_cls):
        checker, _ = self._make_checker()
        result = await checker.acheck_input("")
        assert result.reject is False

    @patch("extensions.guardrail.guardrail_check.Client")
    @patch("extensions.guardrail.guardrail_check.Config")
    async def test_check_input_safe(self, mock_config, mock_client_cls):
        checker, mock_client = self._make_checker()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.body.code = 200
        mock_response.body.data.risk_level = "low"
        mock_response.body.data.advice = []
        mock_response.body.data.result = []
        mock_client.text_moderation_plus_async = AsyncMock(return_value=mock_response)
        checker.client = mock_client

        result = await checker.acheck_input("hello world")
        assert result.reject is False
        assert result.risk_level == "low"

    @patch("extensions.guardrail.guardrail_check.Client")
    @patch("extensions.guardrail.guardrail_check.Config")
    async def test_check_input_high_risk(self, mock_config, mock_client_cls):
        checker, mock_client = self._make_checker()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.body.code = 200
        mock_response.body.data.risk_level = "high"

        mock_advice = MagicMock()
        mock_advice.answer = "Please use appropriate language."
        mock_response.body.data.advice = [mock_advice]

        mock_result_item = MagicMock()
        mock_result_item.description = "Contains harmful content"
        mock_response.body.data.result = [mock_result_item]

        mock_client.text_moderation_plus_async = AsyncMock(return_value=mock_response)
        checker.client = mock_client

        result = await checker.acheck_input("harmful content")
        assert result.reject is True
        assert result.risk_level == "high"
        assert result.advice == "Please use appropriate language."

    @patch("extensions.guardrail.guardrail_check.Client")
    @patch("extensions.guardrail.guardrail_check.Config")
    async def test_check_input_api_error(self, mock_config, mock_client_cls):
        checker, mock_client = self._make_checker()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client.text_moderation_plus_async = AsyncMock(return_value=mock_response)
        checker.client = mock_client

        result = await checker.acheck_input("test")
        assert result.reject is False
        assert result.risk_level == "unknown"

    @patch("extensions.guardrail.guardrail_check.Client")
    @patch("extensions.guardrail.guardrail_check.Config")
    async def test_check_input_exception(self, mock_config, mock_client_cls):
        checker, mock_client = self._make_checker()
        mock_client.text_moderation_plus_async = AsyncMock(side_effect=Exception("network error"))
        checker.client = mock_client

        result = await checker.acheck_input("test")
        assert result.reject is False

    @patch("extensions.guardrail.guardrail_check.Client")
    @patch("extensions.guardrail.guardrail_check.Config")
    async def test_check_output_skips_when_rejected(self, mock_config, mock_client_cls):
        checker, _ = self._make_checker()
        current = TextCheckResult(reject=True, advice="blocked")
        await checker.acheck_output("text", current)
        # Should not make any API call, stays rejected
        assert current.reject is True

    @patch("extensions.guardrail.guardrail_check.Client")
    @patch("extensions.guardrail.guardrail_check.Config")
    async def test_check_output_rejects(self, mock_config, mock_client_cls):
        checker, mock_client = self._make_checker()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.body.code = 200
        mock_response.body.data.risk_level = "high"
        mock_advice = MagicMock()
        mock_advice.answer = "Blocked"
        mock_response.body.data.advice = [mock_advice]
        mock_response.body.data.result = [MagicMock(description="bad")]
        mock_client.text_moderation_plus_async = AsyncMock(return_value=mock_response)
        checker.client = mock_client

        current = TextCheckResult(reject=False)
        await checker.acheck_output("bad output", current)
        assert current.reject is True
