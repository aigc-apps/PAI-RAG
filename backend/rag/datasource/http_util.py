"""Small HTTP helper for data source adapters.

Centralised so adapters share a single User-Agent / timeout policy and so tests
can monkeypatch one function (``http_get``) instead of the network.

SSRF hardening: the host is resolved and validated ONCE, then the request is sent
to that pinned IP while the original Host header / TLS SNI / cert hostname are
preserved. This closes the DNS-rebinding window where a client re-resolves at
connect time and lands on a private address. Env proxies are disabled so a
proxy hop cannot route around the pin. Every redirect hop is re-validated and
re-pinned.

Resource bounds: the response body is streamed with a hard byte cap and a total
read-time deadline so a malicious/slow public host cannot exhaust worker memory
or pin a worker forever (urllib3's timeout is per-read, not total).
"""

import os
import random
import threading
import time
from urllib.parse import urljoin, urlparse

import urllib3
from loguru import logger

from rag.datasource.url_guard import resolve_validated_ip
from utils.constants import try_get_int_env


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")

# A browser-like User-Agent reduces spurious anti-bot challenges from doc hosts
# (e.g. help.aliyun.com). Override via env for honesty or a custom contact UA.
UA = os.environ.get(
    "PAIRAG_DATASOURCE_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)
DEFAULT_TIMEOUT = 30
MAX_REDIRECTS = 5
_REDIRECT_CODES = (301, 302, 303, 307, 308)
_STREAM_CHUNK = 65536

# Hard caps (override via env). A documentation page is well under 10 MB; the time
# bound stops a drip-feeding host from holding a worker past the per-read timeout.
MAX_RESPONSE_BYTES = try_get_int_env("PAIRAG_DATASOURCE_MAX_RESPONSE_BYTES", 10 * 1024 * 1024)
MAX_READ_SECONDS = try_get_int_env("PAIRAG_DATASOURCE_MAX_READ_SECONDS", 60)

# Retry policy for transient failures (connection errors, timeouts, 429/5xx and
# anti-bot challenge pages). Bounded so a worker can't spin forever.
MAX_RETRIES = try_get_int_env("PAIRAG_DATASOURCE_MAX_RETRIES", 3)
_RETRY_BASE_DELAY = 0.5
_RETRY_MAX_DELAY = 10.0
_RETRYABLE_STATUS = (403, 408, 425, 429, 500, 502, 503, 504)

# Opt-in headless-browser fallback: when a challenge survives all retries, run the
# challenge JS in Chromium (Playwright) to complete the cookie handshake, then
# fetch the real body. Requires `pip install playwright && playwright install
# chromium`. Off by default (needs browser binaries; one browser launch per hit).
BROWSER_FALLBACK = _env_flag("PAIRAG_DATASOURCE_BROWSER_FALLBACK")

# Substrings marking an anti-bot / WAF challenge page (e.g. Aliyun x5sec) served
# with HTTP 200 in place of the real content. Treated as a retryable failure so a
# captcha page is never decoded and ingested as document text.
_CHALLENGE_MARKERS = (
    "x5secdata",
    "rgv587_flag",
    '"action":"captcha"',
    "/punish?x5secdata",
)


class FetchLimitExceeded(Exception):
    """Raised when a fetch exceeds a size / time / redirect bound (not retried)."""


class RetryableFetchError(Exception):
    """A transient failure worth retrying (status, transport, or challenge)."""


class ChallengeDetected(RetryableFetchError):
    """The response body is an anti-bot / WAF challenge, not real content."""


class RetryableStatus(RetryableFetchError):
    """A retryable HTTP status (429 / 5xx / 403 …), with optional Retry-After."""

    def __init__(self, status: int, retry_after: float | None = None):
        super().__init__(f"GET failed with retryable status {status}")
        self.status = status
        self.retry_after = retry_after


def _charset(content_type: str) -> str:
    """Pull the charset from a Content-Type header, defaulting to utf-8."""
    if content_type and "charset=" in content_type.lower():
        return content_type.lower().split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
    return "utf-8"


def _read_capped(resp, url: str) -> bytes:
    """Stream the body to memory, aborting if it exceeds the size/time bounds.

    The time bound is a genuine wall-clock deadline: a watchdog closes the
    connection when it elapses, so a slow-drip host that keeps each socket read
    just under the per-read timeout (and would otherwise block forever inside a
    single ``stream()`` iteration) is force-unblocked, not merely checked between
    chunks.
    """
    declared = resp.headers.get("Content-Length")
    if declared is not None:
        try:
            if int(declared) > MAX_RESPONSE_BYTES:
                raise FetchLimitExceeded(
                    f"Response from '{url}' too large: Content-Length {declared} "
                    f"> {MAX_RESPONSE_BYTES} bytes."
                )
        except ValueError:
            pass  # bogus header — fall through to the streaming cap

    deadline = time.monotonic() + MAX_READ_SECONDS
    timed_out = threading.Event()

    def _abort():
        timed_out.set()
        try:
            resp.close()  # closes the underlying socket → unblocks a stuck read
        except Exception:  # noqa: BLE001
            pass

    watchdog = threading.Timer(MAX_READ_SECONDS, _abort)
    watchdog.daemon = True
    watchdog.start()
    buf = bytearray()
    try:
        for chunk in resp.stream(_STREAM_CHUNK, decode_content=True):
            buf += chunk
            if len(buf) > MAX_RESPONSE_BYTES:
                raise FetchLimitExceeded(
                    f"Response from '{url}' exceeds {MAX_RESPONSE_BYTES} bytes; aborting."
                )
            if timed_out.is_set() or time.monotonic() > deadline:
                raise FetchLimitExceeded(
                    f"Reading '{url}' exceeded {MAX_READ_SECONDS}s; aborting."
                )
    except FetchLimitExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        # The watchdog closing the socket surfaces here as an I/O error — translate
        # it to the deadline error so callers see the real cause.
        if timed_out.is_set():
            raise FetchLimitExceeded(
                f"Reading '{url}' exceeded {MAX_READ_SECONDS}s; aborting."
            ) from e
        raise
    finally:
        watchdog.cancel()
    return bytes(buf)


def _fetch_once(url: str, timeout: int):
    """One GET to the pinned, validated IP. Returns (status, headers, body_bytes).

    Redirect responses return an empty body (only the Location header is needed).
    """
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query

    # Validate + pin the address we will actually connect to (anti DNS-rebinding).
    # None => private networks explicitly allowed; resolve normally by hostname.
    pinned_ip = resolve_validated_ip(url)
    connect_host = pinned_ip or host
    headers = {
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": f"{parsed.scheme}://{host}/",
    }
    conn_kw = {}
    if pinned_ip is not None:
        # Connecting by IP: carry the real Host header, and keep TLS SNI + cert
        # verification bound to the original hostname.
        default_port = 443 if parsed.scheme == "https" else 80
        headers["Host"] = host if port == default_port else f"{host}:{port}"
        if parsed.scheme == "https":
            conn_kw = {"server_hostname": host, "assert_hostname": host}

    if parsed.scheme == "https":
        pool = urllib3.HTTPSConnectionPool(
            connect_host, port=port, timeout=timeout, retries=False,
            cert_reqs="CERT_REQUIRED", **conn_kw,
        )
    else:
        pool = urllib3.HTTPConnectionPool(
            connect_host, port=port, timeout=timeout, retries=False,
        )
    try:
        # preload_content=False so we stream with our own caps instead of buffering
        # an unbounded body up front.
        resp = pool.request(
            "GET", target, headers=headers, redirect=False, preload_content=False,
        )
        try:
            if resp.status in _REDIRECT_CODES:
                return resp.status, resp.headers, b""
            return resp.status, resp.headers, _read_capped(resp, url)
        finally:
            resp.release_conn()
    finally:
        pool.close()


def _looks_like_challenge(text: str) -> bool:
    """True if the body is an anti-bot / WAF challenge rather than real content."""
    low = text.lower()
    return any(marker in low for marker in _CHALLENGE_MARKERS)


def _parse_retry_after(headers) -> "float | None":
    """Numeric Retry-After seconds, capped; ignores HTTP-date form."""
    val = headers.get("Retry-After")
    if not val:
        return None
    try:
        return min(float(val), _RETRY_MAX_DELAY)
    except (TypeError, ValueError):
        return None


def _get_following_redirects(url: str, timeout: int) -> str:
    """One attempt: follow redirects, validate status, decode, reject challenges.

    Raises ``RetryableStatus`` / ``ChallengeDetected`` for transient problems and
    ``urllib3.exceptions.HTTPError`` for non-retryable 4xx.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        status, headers, body = _fetch_once(current, timeout)
        if status in _REDIRECT_CODES:
            location = headers.get("Location")
            if not location:
                break
            current = urljoin(current, location)
            continue
        if status in _RETRYABLE_STATUS:
            raise RetryableStatus(status, _parse_retry_after(headers))
        if status >= 400:
            raise urllib3.exceptions.HTTPError(
                f"GET {current} failed with status {status}"
            )
        encoding = _charset(headers.get("Content-Type", ""))
        try:
            text = body.decode(encoding, errors="replace")
        except LookupError:
            text = body.decode("utf-8", errors="replace")
        if _looks_like_challenge(text):
            raise ChallengeDetected(
                f"Anti-bot challenge page returned for '{current}'."
            )
        return text
    raise FetchLimitExceeded(f"Too many redirects while fetching '{url}'.")


# Transient urllib3 transport errors (the base ``HTTPError`` raised for a
# non-retryable 4xx is deliberately NOT listed, so it propagates immediately).
_RETRYABLE_EXC = (
    RetryableFetchError,
    urllib3.exceptions.TimeoutError,
    urllib3.exceptions.ProtocolError,
    urllib3.exceptions.NewConnectionError,
    urllib3.exceptions.MaxRetryError,
)


def _retry_delay(attempt: int, retry_after: "float | None") -> float:
    """Exponential backoff with jitter, or the server's Retry-After if given."""
    if retry_after is not None:
        return retry_after
    backoff = min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY)
    return backoff + random.uniform(0, backoff / 2)


def http_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """GET a URL and return the decoded text body.

    SSRF-guarded: only http/https to public hosts, the connection is pinned to the
    validated IP, every redirect hop is re-validated (auto-redirects disabled and
    followed manually), and the body is bounded in size and read time.

    Transient failures — connection errors, timeouts, 429/5xx statuses, and
    anti-bot challenge pages — are retried with exponential backoff (honouring
    Retry-After). Size/time/redirect bounds and non-retryable 4xx are not retried.
    A challenge that survives all retries raises ``ChallengeDetected`` so the
    caller records an error instead of ingesting the captcha page as content.
    """
    last_exc: Exception = RuntimeError(f"Failed to fetch '{url}'.")
    for attempt in range(MAX_RETRIES + 1):
        try:
            return _get_following_redirects(url, timeout)
        except _RETRYABLE_EXC as e:
            last_exc = e
            if attempt >= MAX_RETRIES:
                break
            time.sleep(_retry_delay(attempt, getattr(e, "retry_after", None)))

    # Anti-bot challenge survived every retry → optionally hand off to a real
    # browser that can run the challenge JS / cookie handshake.
    if BROWSER_FALLBACK and isinstance(last_exc, ChallengeDetected):
        try:
            from rag.datasource import browser_fetch

            logger.warning(f"http_get: persistent challenge on '{url}'; trying browser fallback.")
            return browser_fetch.browser_get(url, timeout)
        except ChallengeDetected:
            raise
        except ImportError:
            logger.warning(
                "Browser fallback is enabled but Playwright is not installed "
                "(`pip install playwright && playwright install chromium`); "
                f"raising the challenge for '{url}'."
            )
        except Exception as be:  # noqa: BLE001
            logger.warning(f"Browser fallback failed for '{url}': {be}")

    raise last_exc
