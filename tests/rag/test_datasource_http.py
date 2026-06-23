"""Tests for the data-source HTTP fetcher's stability hardening.

Covers retry/backoff and anti-bot challenge detection by monkeypatching the
single network primitive (``_fetch_once``) — no real sockets.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import urllib3  # noqa: E402

from rag.datasource import browser_fetch, http_util  # noqa: E402
from rag.datasource.http_util import ChallengeDetected, http_get  # noqa: E402

_CHALLENGE_BODY = (200, {"Content-Type": "text/html"}, b"x5secdata=z;<!--rgv587_flag:sm-->")


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


# --------------------------------------------------------------------------- #
# Playwright browser fallback wiring
# --------------------------------------------------------------------------- #
def test_browser_fallback_off_by_default(monkeypatch):
    monkeypatch.setattr(http_util, "BROWSER_FALLBACK", False)
    calls = {"n": 0}
    monkeypatch.setattr(browser_fetch, "browser_get", lambda *a, **k: calls.__setitem__("n", 1))
    _seq(monkeypatch, [_CHALLENGE_BODY])
    with pytest.raises(ChallengeDetected):
        http_get("https://help.aliyun.com/x.md")
    assert calls["n"] == 0  # browser never invoked when disabled


def test_browser_fallback_used_when_enabled(monkeypatch):
    monkeypatch.setattr(http_util, "BROWSER_FALLBACK", True)
    monkeypatch.setattr(browser_fetch, "browser_get", lambda url, timeout=45: "REAL CONTENT")
    _seq(monkeypatch, [_CHALLENGE_BODY])
    assert http_get("https://help.aliyun.com/x.md") == "REAL CONTENT"


def test_browser_fallback_persistent_challenge_raises(monkeypatch):
    monkeypatch.setattr(http_util, "BROWSER_FALLBACK", True)

    def _boom(url, timeout=45):
        raise ChallengeDetected("still blocked")

    monkeypatch.setattr(browser_fetch, "browser_get", _boom)
    _seq(monkeypatch, [_CHALLENGE_BODY])
    with pytest.raises(ChallengeDetected):
        http_get("https://help.aliyun.com/x.md")


def test_browser_fallback_missing_playwright_raises_original_challenge(monkeypatch):
    monkeypatch.setattr(http_util, "BROWSER_FALLBACK", True)

    def _no_pw(url, timeout=45):
        raise ImportError("No module named 'playwright'")

    monkeypatch.setattr(browser_fetch, "browser_get", _no_pw)
    _seq(monkeypatch, [_CHALLENGE_BODY])
    with pytest.raises(ChallengeDetected):  # original challenge, not ImportError
        http_get("https://help.aliyun.com/x.md")


def test_browser_get_rejects_post_solve_challenge(monkeypatch):
    monkeypatch.setattr(browser_fetch, "validate_public_url", lambda url: None)
    monkeypatch.setattr(browser_fetch, "_run_sync", lambda url, timeout: "x5secdata persists")
    with pytest.raises(ChallengeDetected):
        browser_fetch.browser_get("https://help.aliyun.com/x.md")


def test_browser_get_returns_clean_body(monkeypatch):
    monkeypatch.setattr(browser_fetch, "validate_public_url", lambda url: None)
    monkeypatch.setattr(browser_fetch, "_run_sync", lambda url, timeout: "# Real doc")
    assert browser_fetch.browser_get("https://help.aliyun.com/x.md") == "# Real doc"
