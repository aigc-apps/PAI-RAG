"""Small HTTP helper for data source adapters.

Centralised so adapters share a single User-Agent / timeout policy and so tests
can monkeypatch one function (``http_get``) instead of the network.
"""

from urllib.parse import urljoin

import requests

from rag.datasource.url_guard import validate_public_url, UrlNotAllowed

UA = "Mozilla/5.0 (compatible; pairag-datasource/1.0)"
DEFAULT_TIMEOUT = 30
MAX_REDIRECTS = 5


def http_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """GET a URL and return the decoded text body.

    SSRF-guarded: only http/https to public hosts, and every redirect hop is
    re-validated (auto-redirects are disabled and followed manually).
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        validate_public_url(current)
        resp = requests.get(
            current, headers={"User-Agent": UA}, timeout=timeout, allow_redirects=False
        )
        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location")
            if not location:
                break
            current = urljoin(current, location)
            continue
        resp.raise_for_status()
        if not resp.encoding:
            resp.encoding = "utf-8"
        return resp.text
    raise UrlNotAllowed(f"Too many redirects while fetching '{url}'.")
