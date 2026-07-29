"""Yuque (语雀) adapter — for Yuque knowledge bases.

Pulls documents from a Yuque knowledge base via the Yuque API v2. Supports
filtering by a specific slug to only ingest a subtree of the knowledge base.

discover: fetch the TOC tree from the Yuque API, optionally filtered by a
          starting slug (the node itself is included as the root of the
          subtree). Falls back to listing all docs if the TOC endpoint is
          unavailable.
fetch:    download the document body as raw markdown.

source_config:
    base_url:  API base URL (default: https://yuque-api.antfin-inc.com/api/v2)
    web_url:   Web frontend URL for generating source links
               (default: https://aliyuque.antfin.com)
    namespace: {group}/{repo} — the team (group) and knowledge base (repo) name
               (required)
    slug:      optional starting slug; if given, only that node and its children
               are ingested. If omitted, the entire knowledge base is ingested.
    lang:      optional language tag stored on every doc (default "zh")
"""

import json
import os
import urllib.request
import urllib.error
from typing import List, Optional

from loguru import logger

from rag.datasource.base_adapter import BaseAdapter
from rag.datasource.schema import DiscoveredDoc

YUQUE_DEFAULT_BASE_URL = "https://yuque-api.antfin-inc.com/api/v2"
YUQUE_DEFAULT_WEB_URL = "https://aliyuque.antfin.com"


def _get_token() -> str:
    token = os.environ.get("YUQUE_TOKEN")
    if not token:
        raise ValueError("YUQUE_TOKEN environment variable is not set.")
    return token


def _yuque_api_get(url: str) -> dict:
    """GET a Yuque API endpoint with auth headers; returns parsed JSON."""
    token = _get_token()
    token_suffix = token[-4:] if token else "NONE"
    logger.debug(f"[yuque] GET {url} (token=...{token_suffix})")
    req = urllib.request.Request(url)
    req.add_header("X-Auth-Token", token)
    req.add_header("User-Agent", "pairag-yuque-adapter/1.0")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            status = resp.status
            body = raw.decode("utf-8")
            logger.debug(
                f"[yuque] response {status} len={len(raw)} "
                f"url={url} preview={body[:200]}"
            )
            if not body.strip():
                raise RuntimeError(
                    f"Yuque API returned empty response (status={status}) for {url}. "
                    f"Check YUQUE_TOKEN and network connectivity."
                )
            return json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Yuque API returned non-JSON response for {url}: "
            f"status={status}, body_preview={body[:500]}"
        ) from e
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            f"Yuque API HTTP {e.code} for {url}: {body[:500]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Yuque API unreachable for {url}: {e.reason}"
        ) from e


class YuqueAdapter(BaseAdapter):
    """Discover + fetch documents from a Yuque knowledge base."""

    source_type = "yuque"
    fetched_from = "yuque-api"

    # ------------------------------------------------------------------
    # config helpers
    # ------------------------------------------------------------------
    def _base_url(self) -> str:
        return self.source_config.get("base_url", YUQUE_DEFAULT_BASE_URL).rstrip("/")

    def _namespace(self) -> str:
        ns = self.source_config.get("namespace")
        if not ns:
            raise ValueError(
                "yuque source_config requires 'namespace' (e.g. 'team/repo')."
            )
        return ns

    def _web_url(self) -> str:
        return self.source_config.get("web_url", YUQUE_DEFAULT_WEB_URL).rstrip("/")

    # ------------------------------------------------------------------
    # TOC / discovery
    # ------------------------------------------------------------------
    def _get_toc(self) -> List[dict]:
        """Fetch the full TOC from the Yuque API (flat list with parent_uuid)."""
        base = self._base_url()
        ns = self._namespace()
        url = f"{base}/repos/{ns}/toc"
        resp = _yuque_api_get(url)
        data = resp.get("data", [])
        if not isinstance(data, list):
            raise RuntimeError(
                f"Unexpected TOC response format from {url}: {type(data)}"
            )
        return data

    def _list_all_docs(self) -> List[dict]:
        """Fallback: paginate through the docs list endpoint.

        Returns a flat list whose items mimic the TOC shape (no parent/child
        links — the entire KB is treated as a flat list of root nodes).
        """
        base = self._base_url()
        ns = self._namespace()
        all_docs: List[dict] = []
        offset = 0
        limit = 100
        while True:
            url = f"{base}/repos/{ns}/docs?offset={offset}&limit={limit}&optional=1"
            resp = _yuque_api_get(url)
            data = resp.get("data", [])
            if not data:
                break
            for doc in data:
                all_docs.append({
                    "title": doc.get("title", ""),
                    "slug": doc.get("slug", ""),
                    "uuid": str(doc.get("id", "")),
                    "child_uuid": "",
                    "parent_uuid": "",
                    "depth": 0,
                })
            total = resp.get("meta", {}).get("total", 0) if isinstance(resp.get("meta"), dict) else 0
            offset += limit
            if offset >= total:
                break
        return all_docs

    @staticmethod
    def _build_tree(nodes: List[dict]) -> List[dict]:
        """Build a tree from flat TOC nodes (linked by parent_uuid).

        Each node dict receives two extra keys:
          - ``children``: list of child node dicts
          - ``_parent``:  parent node dict, or None for roots
        """
        # index by uuid
        node_map: dict = {}
        for n in nodes:
            n.setdefault("children", [])
            n["_parent"] = None
            node_map[n["uuid"]] = n

        roots: List[dict] = []
        for n in nodes:
            parent_uuid = n.get("parent_uuid", "")
            if parent_uuid and parent_uuid in node_map:
                parent = node_map[parent_uuid]
                parent["children"].append(n)
                n["_parent"] = parent
            else:
                roots.append(n)
        return roots

    @staticmethod
    def _filter_tree_by_slug(roots: List[dict], slug: str) -> List[dict]:
        """Return the subtree rooted at the node whose slug matches *slug*."""

        def _find(nodes: List[dict]) -> Optional[List[dict]]:
            for n in nodes:
                if n.get("slug") == slug:
                    return [n]
                if n.get("children"):
                    found = _find(n["children"])
                    if found:
                        return found
            return None

        result = _find(roots)
        if not result:
            raise ValueError(
                f"Slug '{slug}' not found in the Yuque knowledge base TOC."
            )
        return result

    @staticmethod
    def _flatten_tree(nodes: List[dict]) -> List[dict]:
        """Flatten tree into a list, storing a ``_section`` field on each node.

        ``_section`` is the ancestor chain joined by `` > ``, or None for roots.
        """
        result: List[dict] = []

        def _walk(node: dict, ancestors: List[dict]):
            section = " > ".join(a["title"] for a in ancestors) if ancestors else None
            node["_section"] = section
            result.append(node)
            for child in node.get("children", []):
                _walk(child, ancestors + [node])

        for root in nodes:
            _walk(root, [])
        return result

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------
    def discover(self) -> List[DiscoveredDoc]:
        """List every document in the (optionally slug-filtered) knowledge base."""
        ns = self._namespace()
        web_url = self._web_url()
        base = self._base_url()
        lang = self.source_config.get("lang", "zh")
        start_slug = self.source_config.get("slug")

        # 1. Get the raw node list (TOC first, fallback to doc listing)
        try:
            toc_nodes = self._get_toc()
        except Exception as e:
            logger.warning(
                f"[yuque] TOC endpoint failed for '{ns}': {e}. "
                f"Falling back to doc listing."
            )
            try:
                toc_nodes = self._list_all_docs()
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to list documents from Yuque knowledge base '{ns}': {e2}"
                ) from e2

        if not toc_nodes:
            raise ValueError(
                f"No documents found in Yuque knowledge base '{ns}'."
            )

        # 2. Build tree
        roots = self._build_tree(toc_nodes)

        # 3. Filter by slug (if specified)
        if start_slug:
            roots = self._filter_tree_by_slug(roots, start_slug)

        # 4. Flatten
        flat = self._flatten_tree(roots)

        # 5. Build DiscoveredDoc list
        docs: List[DiscoveredDoc] = []
        for node in flat:
            slug = node.get("slug", "")
            source_id = str(node.get("uuid", "")).strip()
            if not source_id:
                raise ValueError(
                    f"Yuque document '{slug}' has no UUID and cannot be synchronized safely."
                )
            title = node.get("title", slug)
            page_url = f"{web_url}/{ns}/{slug}"
            # raw=1 returns the body as markdown
            fetch_url = f"{base}/repos/{ns}/docs/{slug}?raw=1"

            docs.append(DiscoveredDoc(
                path=slug,
                title=title,
                source_id=source_id,
                source_url=page_url,
                fetch_url=fetch_url,
                section=node.get("_section"),
                summary=None,
                product=None,
                lang=lang,
                source_meta={
                    "depth": node.get("depth", 0),
                    "uuid": node.get("uuid", ""),
                },
            ))

        logger.info(
            f"[yuque] discovered {len(docs)} docs from '{ns}'"
            + (f" (slug='{start_slug}')" if start_slug else "")
        )
        return docs

    def fetch(self, doc: DiscoveredDoc) -> str:
        """Return the raw markdown body for a document."""
        resp = _yuque_api_get(doc.fetch_url)
        data = resp.get("data", {})
        # When raw=1, data can be a string (the body) or a dict with "body" key
        if isinstance(data, dict):
            body = data.get("body", "")
        elif isinstance(data, str):
            body = data
        else:
            body = ""
        if not body:
            logger.warning(f"[yuque] empty body for {doc.path}")
        return body
