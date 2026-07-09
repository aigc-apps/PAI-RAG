"""SSRF guard for adapter-issued HTTP requests.

User-supplied data source URLs (llms_url / product slug) are fetched server-side,
so they must be constrained: only http/https, and the host must not resolve to a
private / loopback / link-local / reserved address. Redirects are validated too
(see http_util.http_get).

Set PAIRAG_DATASOURCE_ALLOW_PRIVATE_NETWORK=true to permit private hosts (e.g. an
internal documentation site on a corporate network).

Ported verbatim from backend/rag/datasource/url_guard.py (validate_public_url path).
"""

import os
import ipaddress
import socket
from urllib.parse import urlparse


class UrlNotAllowed(ValueError):
    """Raised when a URL is rejected by the SSRF guard."""


def _allow_private() -> bool:
    return os.environ.get("PAIRAG_DATASOURCE_ALLOW_PRIVATE_NETWORK", "false").lower() == "true"


def _is_blocked_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparsable → treat as blocked
    return (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


def validate_public_url(url: str) -> None:
    """Raise UrlNotAllowed unless `url` is http/https to a public host.

    Resolves the host and rejects when ANY returned record is non-public (a mixed
    answer could otherwise slip past a check that only inspects the first record).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UrlNotAllowed(f"Blocked URL scheme '{parsed.scheme}': only http/https are allowed.")
    host = parsed.hostname
    if not host:
        raise UrlNotAllowed(f"Blocked URL '{url}': missing host.")
    if _allow_private():
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise UrlNotAllowed(f"Cannot resolve host '{host}': {e}") from e
    for info in infos:
        ip = info[4][0]
        if _is_blocked_ip(ip):
            raise UrlNotAllowed(
                f"Blocked URL '{url}': host '{host}' resolves to non-public address {ip}."
            )
