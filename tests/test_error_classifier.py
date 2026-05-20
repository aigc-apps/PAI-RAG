"""Unit tests for error_classifier."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from error_classifier import (
    FailureReason,
    classify_api_error,
)


# Synthetic exception types: real openai SDK exceptions vary in shape across
# versions; testing against a class hierarchy we control avoids version drift
# and exercises the same attribute paths the classifier walks.

class FakeResponse:
    def __init__(self, status_code=None, text=''):
        self.status_code = status_code
        self.text = text


class FakeAPIError(Exception):
    def __init__(self, message='', status_code=None, body=None, response=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body
        self.response = response


# ────────────── Status-code → reason ────────────── #

def test_401_is_auth_retryable_with_rotate():
    c = classify_api_error(FakeAPIError('bad key', status_code=401))
    assert c.reason == FailureReason.AUTH
    assert c.retryable is True
    assert c.should_rotate_credential is True
    assert c.should_compress is False


def test_403_is_auth_permanent_no_retry():
    c = classify_api_error(FakeAPIError('forbidden', status_code=403))
    assert c.reason == FailureReason.AUTH_PERMANENT
    assert c.retryable is False
    assert c.should_rotate_credential is True


def test_429_is_rate_limit():
    c = classify_api_error(FakeAPIError('slow down', status_code=429))
    assert c.reason == FailureReason.RATE_LIMIT
    assert c.retryable is True
    assert c.should_rotate_credential is True


def test_503_is_overloaded():
    c = classify_api_error(FakeAPIError('overloaded', status_code=503))
    assert c.reason == FailureReason.OVERLOADED
    assert c.retryable is True
    assert c.should_rotate_credential is False


def test_500_is_server_error():
    c = classify_api_error(FakeAPIError('boom', status_code=500))
    assert c.reason == FailureReason.SERVER_ERROR
    assert c.retryable is True


def test_400_context_overflow_via_text():
    c = classify_api_error(
        FakeAPIError('This model maximum context length is 8192 tokens',
                     status_code=400)
    )
    assert c.reason == FailureReason.CONTEXT_OVERFLOW
    assert c.should_compress is True
    assert c.retryable is True


def test_400_plain_bad_request_not_retryable():
    c = classify_api_error(FakeAPIError('invalid value', status_code=400))
    assert c.reason == FailureReason.BAD_REQUEST
    assert c.retryable is False
    assert c.should_compress is False


# ────────────── Cause-chain walk ────────────── #

def test_status_extracted_from_cause_chain():
    inner = FakeAPIError('rate limited', status_code=429)
    outer = RuntimeError('wrapper')
    outer.__cause__ = inner
    c = classify_api_error(outer)
    assert c.reason == FailureReason.RATE_LIMIT
    assert c.status_code == 429


def test_status_extracted_from_response_attribute():
    exc = FakeAPIError('server unhappy', response=FakeResponse(status_code=502))
    # Note: status_code on the exception itself is None
    c = classify_api_error(exc)
    assert c.reason == FailureReason.SERVER_ERROR
    assert c.status_code == 502


def test_cause_chain_loop_does_not_hang():
    a = RuntimeError('a')
    b = RuntimeError('b')
    a.__cause__ = b
    b.__cause__ = a
    # If the walk doesn't dedupe, this hangs forever.
    c = classify_api_error(a)
    assert c.reason == FailureReason.UNKNOWN


# ────────────── OpenRouter unwrap ────────────── #

def test_openrouter_unwrap_recovers_inner_429():
    # Outer code is 200 (or even missing) but the real upstream said 429.
    body = json.dumps({
        'error': {
            'code': 200,
            'metadata': {
                'raw': json.dumps({'error': {'code': 429, 'message': 'rate limited'}})
            },
        }
    })
    exc = FakeAPIError('upstream said something', body=body)
    c = classify_api_error(exc)
    assert c.reason == FailureReason.RATE_LIMIT
    assert c.status_code == 429


def test_openrouter_unwrap_skips_when_no_inner():
    body = json.dumps({'error': {'code': 500, 'message': 'oops'}})
    exc = FakeAPIError('outer', status_code=500, body=body)
    c = classify_api_error(exc)
    assert c.reason == FailureReason.SERVER_ERROR
    assert c.status_code == 500


# ────────────── Text-pattern fallback ────────────── #

def test_timeout_text_without_status():
    c = classify_api_error(TimeoutError('Request timed out after 30s'))
    assert c.reason == FailureReason.TIMEOUT


def test_overloaded_text_without_status():
    c = classify_api_error(RuntimeError('Anthropic API: server overloaded'))
    assert c.reason == FailureReason.OVERLOADED


def test_context_overflow_text_without_status():
    c = classify_api_error(RuntimeError(
        'Your prompt is too long; reduce the length and try again'))
    assert c.reason == FailureReason.CONTEXT_OVERFLOW
    assert c.should_compress is True


def test_unknown_falls_back_retryable():
    c = classify_api_error(RuntimeError('???'))
    assert c.reason == FailureReason.UNKNOWN
    assert c.retryable is True
    assert c.should_rotate_credential is False


def test_none_input_safe():
    c = classify_api_error(None)
    assert c.reason == FailureReason.UNKNOWN
