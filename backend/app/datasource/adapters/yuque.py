"""Yuque (语雀) adapter — sync one Yuque knowledge base (a *book*) into a KB.

A Yuque book is a document tree; its TOC lists TITLE / DOC / LINK nodes linked by
``parent_uuid``. This adapter walks the TOC, optionally narrows to a sub-tree
(``path`` / ``roots``), and fetches each DOC's markdown body via the Yuque
OpenAPI. Only two read endpoints are used: ``/repos/{group}/{book}/toc`` and
``/repos/{group}/{book}/docs/{slug}``.

discover: fetch the TOC, index by uuid, resolve the in-scope uuid set (whole book
          when no path/roots given), and emit one DiscoveredDoc per DOC node.
fetch:    GET the doc detail and return ``data.body`` (markdown source).

source_config:
    group_login: Yuque space / group login (namespace owner)   [required]
    book_slug:   the book's slug                               [required]
    token_env:   name of the env var holding the X-Auth-Token  [required]
                 (the token itself is NEVER stored — source_config is returned
                 to clients; only the env-var *name* lives here)
    api_base:    OpenAPI base, default https://www.yuque.com/api/v2
    web_base:    site base for citation URLs, default derived from api_base
    path:        optional — a single TOC node's slug OR title; syncs its subtree
    roots:       optional — advanced multi-select: [{uuid|slug|title}], each
                 selecting a subtree (including the root node itself)

Reference logic ported from a standalone yuque-export script and trimmed to the
backend's discover/fetch/emit model (no topic-mapping / frontmatter / local state).

Note: the sync loop deletes documents that exist in the KB but are out of this
run's scope. Narrowing ``path`` / ``roots`` between runs therefore removes the
now-out-of-scope docs (expected, but worth knowing).
"""

import os
import threading
import time
from typing import Dict, List, Optional
from urllib.parse import urlparse

import httpx
from loguru import logger

from app.datasource.base_adapter import BaseAdapter
from app.datasource.schema import DiscoveredDoc
from app.datasource.http_util import (
    UA, DEFAULT_TIMEOUT, MAX_REDIRECTS, MAX_READ_SECONDS,
    _REDIRECT_CODES, _read_capped, FetchLimitExceeded,
)
from app.datasource.url_guard import validate_public_url

DEFAULT_API_BASE = "https://www.yuque.com/api/v2"

# Yuque rate-limits the OpenAPI; keep a conservative global floor between calls
# (the sync loop fetches concurrently across threads). A module-level lock +
# monotonic clock enforces a minimum inter-request gap process-wide.
_MIN_INTERVAL = 0.05  # ≤ 20 req/s
_rate_lock = threading.Lock()
_last_request = 0.0


def yuque_get_json(url: str, token: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """GET a Yuque OpenAPI URL with X-Auth-Token and return the parsed ``data``.

    SSRF-guarded like ``http_util.http_get`` (only http/https to public hosts,
    every redirect hop re-validated, env proxies disabled, body size/time
    bounded), plus a process-wide request throttle. Module-level so tests
    monkeypatch this one function instead of the network.
    """
    global _last_request
    with _rate_lock:
        gap = time.monotonic() - _last_request
        if gap < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - gap)
        _last_request = time.monotonic()

    headers = {"User-Agent": UA, "X-Auth-Token": token,
               "Content-Type": "application/json"}
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
                body = _read_capped(resp, current)
                if resp.status_code >= 400:
                    detail = body.decode("utf-8", errors="replace")[:500]
                    raise httpx.HTTPError(
                        f"GET {current} failed with status {resp.status_code}: {detail}"
                    )
                payload = httpx.Response(200, content=body).json()
                return payload.get("data", {}) if isinstance(payload, dict) else {}
    raise FetchLimitExceeded(f"Too many redirects while fetching '{url}'.")


# --------------------------------------------------------------------------- #
# TOC traversal (parent_uuid links) + subtree selection
# --------------------------------------------------------------------------- #
def _build_index(toc: List[dict]) -> Dict[str, dict]:
    return {item["uuid"]: item for item in toc if item.get("uuid")}


def _children_map(toc: List[dict]) -> Dict[str, List[dict]]:
    cm: Dict[str, List[dict]] = {}
    for node in toc:
        cm.setdefault(node.get("parent_uuid"), []).append(node)
    return cm


def _ancestor_titles(node: dict, by_uuid: Dict[str, dict]) -> List[str]:
    """Titles from the node up to the root (node's own title first)."""
    titles = [node.get("title", "")]
    parent_uuid = node.get("parent_uuid")
    seen = set()
    while parent_uuid and parent_uuid in by_uuid and parent_uuid not in seen:
        seen.add(parent_uuid)
        parent = by_uuid[parent_uuid]
        titles.append(parent.get("title", ""))
        parent_uuid = parent.get("parent_uuid")
    return titles


def _match_root(toc: List[dict], spec: dict) -> Optional[dict]:
    """Locate one node by uuid / slug / title. A spec with key ``any`` matches
    against uuid OR slug OR title (used for the convenience ``path`` string)."""
    any_val = spec.get("any")
    for node in toc:
        if spec.get("uuid") and node.get("uuid") == spec["uuid"]:
            return node
        if spec.get("slug") and node.get("slug") == spec["slug"]:
            return node
        if spec.get("title") and node.get("title") == spec["title"]:
            return node
        if any_val and any_val in (node.get("uuid"), node.get("slug"), node.get("title")):
            return node
    return None


def _selected_uuids(toc: List[dict], roots: List[dict]) -> Optional[set]:
    """Uuid set of every root's subtree (including the root). ``None`` (=whole
    book) when ``roots`` is empty."""
    if not roots:
        return None
    cmap = _children_map(toc)
    selected: set = set()
    for spec in roots:
        root = _match_root(toc, spec)
        if not root:
            logger.warning(f"[yuque] path/root not found in TOC: {spec}")
            continue
        stack = [root]
        while stack:
            n = stack.pop()
            uid = n.get("uuid")
            if uid in selected:
                continue
            selected.add(uid)
            stack.extend(cmap.get(uid, []))
    return selected


class YuqueAdapter(BaseAdapter):
    source_type = "yuque"
    fetched_from = "yuque-openapi"

    # -- config -------------------------------------------------------------
    def validate_config(self) -> None:
        cfg = self.source_config
        for key in ("group_login", "book_slug", "token_env"):
            if not str(cfg.get(key) or "").strip():
                raise ValueError(f"yuque source_config requires '{key}'.")
        token_env = cfg["token_env"].strip()
        if not os.environ.get(token_env):
            raise ValueError(
                f"yuque token_env '{token_env}' is not set in the server "
                f"environment (set it to a valid Yuque X-Auth-Token)."
            )

    def _api_base(self) -> str:
        return str(self.source_config.get("api_base") or DEFAULT_API_BASE).rstrip("/")

    def _web_base(self) -> str:
        web = str(self.source_config.get("web_base") or "").rstrip("/")
        if web:
            return web
        parsed = urlparse(self._api_base())
        return f"{parsed.scheme}://{parsed.netloc}"

    def _token(self) -> str:
        token_env = str(self.source_config.get("token_env") or "").strip()
        token = os.environ.get(token_env) if token_env else None
        if not token:
            raise ValueError(
                f"yuque token_env '{token_env}' is not set in the server environment."
            )
        return token

    def _get(self, endpoint: str) -> dict:
        return yuque_get_json(self._api_base() + endpoint, self._token())

    def _roots(self) -> List[dict]:
        """Normalize path/roots config into a list of match specs."""
        cfg = self.source_config
        roots = list(cfg.get("roots") or [])
        path = str(cfg.get("path") or "").strip()
        if path:
            roots.append({"any": path})
        return roots

    # -- stages -------------------------------------------------------------
    def discover(self) -> List[DiscoveredDoc]:
        cfg = self.source_config
        group_login, book_slug = cfg["group_login"], cfg["book_slug"]
        web_base = self._web_base()

        toc = self._get(f"/repos/{group_login}/{book_slug}/toc") or []
        if not isinstance(toc, list):
            raise ValueError(
                f"Unexpected Yuque TOC response for {group_login}/{book_slug}."
            )
        by_uuid = _build_index(toc)
        selected = _selected_uuids(toc, self._roots())

        docs: List[DiscoveredDoc] = []
        for node in toc:
            if node.get("type") != "DOC" or not node.get("slug"):
                continue
            if selected is not None and node.get("uuid") not in selected:
                continue
            slug = node["slug"]
            parent_uuid = node.get("parent_uuid")
            section = by_uuid.get(parent_uuid, {}).get("title") if parent_uuid else None
            page_url = f"{web_base}/{group_login}/{book_slug}/{slug}"
            docs.append(DiscoveredDoc(
                path=slug,
                title=node.get("title", ""),
                source_url=page_url,
                fetch_url=page_url,
                section=section or None,
                source_meta={"uuid": node.get("uuid"), "parent_uuid": parent_uuid},
            ))
        scope = "whole book" if selected is None else f"{len(selected)} nodes in scope"
        logger.info(
            f"[yuque] discovered {len(docs)} docs from {group_login}/{book_slug} ({scope})"
        )
        return docs

    def fetch(self, doc: DiscoveredDoc) -> str:
        cfg = self.source_config
        data = self._get(
            f"/repos/{cfg['group_login']}/{cfg['book_slug']}/docs/{doc.path}"
        )
        return data.get("body") or ""
