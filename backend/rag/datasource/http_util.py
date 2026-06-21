"""Small HTTP helper for data source adapters.

Centralised so adapters share a single User-Agent / timeout policy and so tests
can monkeypatch one function (``http_get``) instead of the network.

SSRF hardening: the host is resolved and validated ONCE, then the request is sent
to that pinned IP while the original Host header / TLS SNI / cert hostname are
preserved. This closes the DNS-rebinding window where a client re-resolves at
connect time and lands on a private address. Env proxies are disabled so a
proxy hop cannot route around the pin. Every redirect hop is re-validated and
re-pinned.
"""

from urllib.parse import urljoin, urlparse

import urllib3

from rag.datasource.url_guard import resolve_validated_ip, UrlNotAllowed

UA = "Mozilla/5.0 (compatible; pairag-datasource/1.0)"
DEFAULT_TIMEOUT = 30
MAX_REDIRECTS = 5
_REDIRECT_CODES = (301, 302, 303, 307, 308)


def _charset(content_type: str) -> str:
    """Pull the charset from a Content-Type header, defaulting to utf-8."""
    if content_type and "charset=" in content_type.lower():
        return content_type.lower().split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
    return "utf-8"


def _request_once(url: str, timeout: int):
    """Issue one GET to the pinned, validated IP. Returns the urllib3 response."""
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
        return pool.request(
            "GET", target, headers=headers, redirect=False, preload_content=True,
        )
    finally:
        pool.close()


def http_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """GET a URL and return the decoded text body.

    SSRF-guarded: only http/https to public hosts, the connection is pinned to the
    validated IP, and every redirect hop is re-validated (auto-redirects disabled
    and followed manually).
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        resp = _request_once(current, timeout)
        if resp.status in _REDIRECT_CODES:
            location = resp.headers.get("Location")
            if not location:
                break
            current = urljoin(current, location)
            continue
        if resp.status >= 400:
            raise urllib3.exceptions.HTTPError(
                f"GET {current} failed with status {resp.status}"
            )
        encoding = _charset(resp.headers.get("Content-Type", ""))
        try:
            return resp.data.decode(encoding, errors="replace")
        except LookupError:
            return resp.data.decode("utf-8", errors="replace")
    raise UrlNotAllowed(f"Too many redirects while fetching '{url}'.")
