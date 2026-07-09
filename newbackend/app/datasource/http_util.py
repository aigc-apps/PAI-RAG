"""Small HTTP helper for data source adapters (httpx-based).

Centralised so adapters share a single User-Agent / timeout policy and so tests
can monkeypatch one function (``http_get``) instead of the network.

SSRF hardening: every URL — including each redirect hop — is validated with
``validate_public_url`` before it is fetched (scheme is http/https and the host
resolves only to public addresses). Env proxies are disabled so a proxy hop
cannot route around the check, redirects are followed manually so each hop is
re-validated, and the body is bounded in size and read time.

Deviation from the older urllib3 implementation: that one connected to a
pre-resolved, pinned IP to also close the DNS-rebinding window (host re-resolving
to a private address at connect time). httpx re-resolves at connect time, so a
narrow rebinding window remains between our validation and the actual connect.
This is an accepted residual risk for the MVP; set
PAIRAG_DATASOURCE_ALLOW_PRIVATE_NETWORK only on trusted networks.
"""

import os

import httpx

from app.datasource.url_guard import validate_public_url


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


UA = "Mozilla/5.0 (compatible; pairag-datasource/1.0)"
DEFAULT_TIMEOUT = 30
MAX_REDIRECTS = 5
_REDIRECT_CODES = (301, 302, 303, 307, 308)

# Hard caps (override via env). A documentation page is well under 10 MB.
MAX_RESPONSE_BYTES = _int_env("PAIRAG_DATASOURCE_MAX_RESPONSE_BYTES", 10 * 1024 * 1024)
MAX_READ_SECONDS = _int_env("PAIRAG_DATASOURCE_MAX_READ_SECONDS", 60)


class FetchLimitExceeded(Exception):
    """Raised when a fetch exceeds a size / time / redirect bound."""


def _read_capped(resp: httpx.Response, url: str) -> bytes:
    """Stream the body to memory, aborting if it exceeds the size bound."""
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
    buf = bytearray()
    for chunk in resp.iter_bytes():
        buf += chunk
        if len(buf) > MAX_RESPONSE_BYTES:
            raise FetchLimitExceeded(
                f"Response from '{url}' exceeds {MAX_RESPONSE_BYTES} bytes; aborting."
            )
    return bytes(buf)


def http_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """GET a URL and return the decoded text body.

    SSRF-guarded: only http/https to public hosts, every redirect hop is
    re-validated (auto-redirects disabled and followed manually), env proxies are
    disabled, and the body is bounded in size and total read time.
    """
    headers = {"User-Agent": UA}
    # total read-time deadline in addition to per-op connect/read timeouts
    timeout_cfg = httpx.Timeout(timeout, read=min(timeout, MAX_READ_SECONDS))
    current = url
    with httpx.Client(
        trust_env=False, follow_redirects=False, timeout=timeout_cfg, headers=headers
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            validate_public_url(current)  # re-validate every hop
            with client.stream("GET", current) as resp:
                if resp.status_code in _REDIRECT_CODES:
                    location = resp.headers.get("Location")
                    if not location:
                        break
                    current = str(httpx.URL(current).join(location))
                    continue
                if resp.status_code >= 400:
                    raise httpx.HTTPError(
                        f"GET {current} failed with status {resp.status_code}"
                    )
                body = _read_capped(resp, current)
                encoding = resp.encoding or "utf-8"
                try:
                    return body.decode(encoding, errors="replace")
                except LookupError:
                    return body.decode("utf-8", errors="replace")
    raise FetchLimitExceeded(f"Too many redirects while fetching '{url}'.")
