"""Tests for the data-source HTTP fetcher's stability hardening.

Covers retry/backoff and anti-bot challenge detection by monkeypatching the
single network primitive (``_fetch_once``) — no real sockets.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import urllib3  # noqa: E402

from rag.datasource import http_util  # noqa: E402
from rag.datasource.http_util import ChallengeDetected, http_get  # noqa: E402


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    # No real sleeping; deterministic, small retry budget.
    monkeypatch.setattr(http_util.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(http_util, "MAX_RETRIES", 2)


def _seq(monkeypatch, responses):
    """Stub _fetch_once with a sequence; the last entry repeats.

    Each entry is either a (status, headers, body_bytes) tuple or an Exception
    instance to raise.
    """
    calls = {"n": 0}

    def fake(url, timeout):
        item = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(http_util, "_fetch_once", fake)
    return calls


def test_success_returns_decoded_text(monkeypatch):
    _seq(monkeypatch, [(200, {"Content-Type": "text/html; charset=utf-8"}, b"hello")])
    assert http_get("https://example.com/doc") == "hello"


def test_challenge_page_is_retried_then_raises(monkeypatch):
    body = b"<script>sessionStorage.x5referer=1;x5secdata=abc;</script><!--rgv587_flag:sm-->"
    calls = _seq(monkeypatch, [(200, {"Content-Type": "text/html"}, body)])
    with pytest.raises(ChallengeDetected):
        http_get("https://help.aliyun.com/zh/x.md")
    assert calls["n"] == 3  # MAX_RETRIES (2) + 1 — never returned as content


def test_retryable_status_then_success(monkeypatch):
    calls = _seq(
        monkeypatch,
        [
            (503, {"Retry-After": "0"}, b""),
            (200, {"Content-Type": "text/plain"}, b"ok"),
        ],
    )
    assert http_get("https://example.com/x") == "ok"
    assert calls["n"] == 2


def test_non_retryable_404_raises_immediately(monkeypatch):
    calls = _seq(monkeypatch, [(404, {}, b"")])
    with pytest.raises(urllib3.exceptions.HTTPError):
        http_get("https://example.com/missing")
    assert calls["n"] == 1  # not retried


def test_transport_error_is_retried_then_succeeds(monkeypatch):
    calls = _seq(
        monkeypatch,
        [
            urllib3.exceptions.ProtocolError("connection reset"),
            (200, {"Content-Type": "text/plain"}, b"recovered"),
        ],
    )
    assert http_get("https://example.com/x") == "recovered"
    assert calls["n"] == 2


def test_size_limit_is_not_retried(monkeypatch):
    calls = _seq(monkeypatch, [http_util.FetchLimitExceeded("too big")])
    with pytest.raises(http_util.FetchLimitExceeded):
        http_get("https://example.com/big")
    assert calls["n"] == 1


def test_looks_like_challenge_markers():
    assert http_util._looks_like_challenge('{"action":"captcha"}')
    assert http_util._looks_like_challenge("...X5SECDATA...")  # case-insensitive
    assert not http_util._looks_like_challenge("# A normal doc mentioning captcha bypass")
