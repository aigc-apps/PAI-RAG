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

import time
from urllib.parse import urljoin, urlparse

import urllib3

from rag.datasource.url_guard import resolve_validated_ip
from utils.constants import try_get_int_env

UA = "Mozilla/5.0 (compatible; pairag-datasource/1.0)"
DEFAULT_TIMEOUT = 30
MAX_REDIRECTS = 5
_REDIRECT_CODES = (301, 302, 303, 307, 308)
_STREAM_CHUNK = 65536

# Hard caps (override via env). A documentation page is well under 10 MB; the time
# bound stops a drip-feeding host from holding a worker past the per-read timeout.
MAX_RESPONSE_BYTES = try_get_int_env("PAIRAG_DATASOURCE_MAX_RESPONSE_BYTES", 10 * 1024 * 1024)
MAX_READ_SECONDS = try_get_int_env("PAIRAG_DATASOURCE_MAX_READ_SECONDS", 60)


class FetchLimitExceeded(Exception):
    """Raised when a fetch exceeds a size / time / redirect bound."""


def _charset(content_type: str) -> str:
    """Pull the charset from a Content-Type header, defaulting to utf-8."""
    if content_type and "charset=" in content_type.lower():
        return content_type.lower().split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
    return "utf-8"


def _read_capped(resp, url: str) -> bytes:
    """Stream the body to memory, aborting if it exceeds the size/time bounds."""
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
    buf = bytearray()
    for chunk in resp.stream(_STREAM_CHUNK, decode_content=True):
        buf += chunk
        if len(buf) > MAX_RESPONSE_BYTES:
            raise FetchLimitExceeded(
                f"Response from '{url}' exceeds {MAX_RESPONSE_BYTES} bytes; aborting."
            )
        if time.monotonic() > deadline:
            raise FetchLimitExceeded(
                f"Reading '{url}' exceeded {MAX_READ_SECONDS}s; aborting."
            )
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
    headers = {"User-Agent": UA}
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


def http_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """GET a URL and return the decoded text body.

    SSRF-guarded: only http/https to public hosts, the connection is pinned to the
    validated IP, every redirect hop is re-validated (auto-redirects disabled and
    followed manually), and the body is bounded in size and read time.
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
        if status >= 400:
            raise urllib3.exceptions.HTTPError(
                f"GET {current} failed with status {status}"
            )
        encoding = _charset(headers.get("Content-Type", ""))
        try:
            return body.decode(encoding, errors="replace")
        except LookupError:
            return body.decode("utf-8", errors="replace")
    raise FetchLimitExceeded(f"Too many redirects while fetching '{url}'.")
