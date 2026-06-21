"""Sphinx / readthedocs adapter — for sphinx_rtd_theme sites without an llms.txt.

Ported from docs/agent/data_source/torcheasyrec_docs/fetch_sphinx_docs.py.

These sites have no authoritative manifest, so discovery is a BFS: seed from the
sidebar toctree, then follow in-scope `.html` links found in each page's body.
Because BFS must fetch every page anyway, ``discover()`` renders each page to
markdown and caches the body; ``fetch()`` is then an O(1) cache lookup.

source_config:
    base_url:  homepage URL with the lang/version segment (required), e.g.
               "https://easyrec.readthedocs.io/en/latest/"
    product:   product name (optional)
    site:      source site (optional; defaults to base_url host)
    lang:      language tag stored on every doc (optional)
    workers:   BFS fetch concurrency (optional, default 6)
"""

import re
from typing import List, Optional, Set, Tuple
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
from markdownify import markdownify as html_to_md
from loguru import logger

from rag.datasource.base_adapter import BaseAdapter
from rag.datasource.schema import DiscoveredDoc
from rag.datasource.http_util import http_get
from utils.constants import try_get_int_env

# Sphinx-generated non-content pages / asset dirs we never ingest
EXCLUDE_BASENAMES = {"genindex.html", "search.html", "py-modindex.html", "modindex.html"}
EXCLUDE_PATH_PARTS = ("/_modules/", "/_sources/", "/_static/", "/_downloads/", "/_images/")

# Resource bounds for the BFS crawl (override via env). Cap concurrency so a
# misconfigured `workers` can't open hundreds of sockets, and cap total pages so
# a sprawling / link-looping site can't crawl unbounded.
DEFAULT_WORKERS = 6
MAX_WORKERS = try_get_int_env("PAIRAG_DATASOURCE_SPHINX_MAX_WORKERS", 8)
MAX_PAGES = try_get_int_env("PAIRAG_DATASOURCE_SPHINX_MAX_PAGES", 2000)
# Cumulative cap on rendered-body memory held in _rendered across one crawl
# (per-page + page-count caps alone still allow MAX_PAGES large bodies to pile up).
MAX_TOTAL_BYTES = try_get_int_env("PAIRAG_DATASOURCE_SPHINX_MAX_TOTAL_BYTES", 200 * 1024 * 1024)


def parse_toctree(home_html: str) -> List[Tuple[str, str, str]]:
    """Sidebar toctree → [(href, section, link_text)] as BFS seeds."""
    soup = BeautifulSoup(home_html, "html.parser")
    nav = soup.find("div", class_="wy-menu") or soup
    out, seen, current = [], set(), ""
    for el in nav.descendants:
        name = getattr(el, "name", None)
        if name == "p" and "caption" in (el.get("class") or []):
            current = el.get_text(strip=True)
        elif name == "a" and "internal" in (el.get("class") or []):
            href = (el.get("href") or "").split("#")[0]
            if href.endswith(".html") and not href.startswith("http") and href not in seen:
                seen.add(href)
                out.append((href, current, el.get_text(strip=True)))
    return out


def _in_scope(url: str, base: str) -> bool:
    if not url.startswith(base) or not url.endswith(".html"):
        return False
    if any(p in url for p in EXCLUDE_PATH_PARTS):
        return False
    return url.rsplit("/", 1)[-1] not in EXCLUDE_BASENAMES


def render(html: str, page_url: str, base: str) -> Tuple[Optional[str], Optional[str], Set[str]]:
    """Extract the article body → (title, markdown, in-scope outlinks). Links absolutized."""
    soup = BeautifulSoup(html, "html.parser")
    body = (soup.find("div", attrs={"itemprop": "articleBody"})
            or soup.find("div", attrs={"role": "main"}))
    if body is None:
        body = soup.body
        if body is None:
            return None, None, set()
        for tag in body.select(
            "nav, header, footer, script, style, .wy-nav-side, .rst-footer-buttons, .rst-versions"
        ):
            tag.decompose()
    for a in body.select("a.headerlink"):
        a.decompose()
    for a in body.find_all("a", href=True):
        a["href"] = urljoin(page_url, a["href"])
    for img in body.find_all("img", src=True):
        img["src"] = urljoin(page_url, img["src"])
    outlinks = {
        a["href"].split("#")[0]
        for a in body.find_all("a", href=True)
        if _in_scope(a["href"].split("#")[0], base)
    }
    h1 = body.find("h1")
    title = h1.get_text(strip=True).rstrip("¶").strip() if h1 else None
    markdown = html_to_md(
        str(body), heading_style="ATX", bullets="-", code_language="",
        escape_asterisks=False, escape_underscores=False,
    )
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return title, markdown, outlinks


def _url_to_path(url: str) -> str:
    """Page URL → relative .md path (preserving <lang>/<version>/... structure)."""
    rel = urlparse(url).path.lstrip("/")
    if rel.endswith(".html"):
        return rel[:-5] + ".md"
    return rel.rstrip("/") + "/index.md"


class SphinxAdapter(BaseAdapter):
    source_type = "sphinx"
    fetched_from = "sphinx-html"

    def __init__(self, datasource_key: str, source_config: Optional[dict] = None):
        super().__init__(datasource_key, source_config)
        # path -> (title, markdown, source_url, section) populated by discover()
        self._rendered: dict = {}

    def _base_url(self) -> str:
        base = self.source_config.get("base_url")
        if not base:
            raise ValueError("sphinx source_config requires 'base_url'.")
        return base if base.endswith("/") else base + "/"

    def discover(self) -> List[DiscoveredDoc]:
        base = self._base_url()
        product = self.source_config.get("product") or None
        lang = self.source_config.get("lang") or None
        # Clamp concurrency into [1, MAX_WORKERS] regardless of config.
        workers = max(1, min(int(self.source_config.get("workers", DEFAULT_WORKERS)), MAX_WORKERS))

        seeds = parse_toctree(http_get(base))
        section_of = {urljoin(base, "index.html"): ""}
        for href, section, _ in seeds:
            u = urljoin(base, href).split("#")[0]
            section_of.setdefault(u, section)

        def _render_page(url):
            title, md, links = render(http_get(url), url, base)
            return url, title, md, links

        def _render_slice(urls):
            """Render up to `workers` pages; bounds bodies held in memory at once."""
            out = []
            if workers <= 1 or len(urls) == 1:
                for u in urls:
                    try:
                        out.append(_render_page(u))
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"[sphinx] render failed {u}: {e}")
                        self.discovery_partial = True  # incomplete listing → block deletes
            else:
                with ThreadPoolExecutor(max_workers=min(workers, len(urls))) as ex:
                    futs = {ex.submit(_render_page, u): u for u in urls}
                    for fu in as_completed(futs):
                        try:
                            out.append(fu.result())
                        except Exception as e:  # noqa: BLE001
                            logger.warning(f"[sphinx] render failed {futs[fu]}: {e}")
                            self.discovery_partial = True  # incomplete listing → block deletes
            return out

        visited: Set[str] = set()
        frontier = list(section_of)
        rounds = 0
        total_bytes = 0  # cumulative size of rendered bodies held in _rendered
        stop = False
        while frontier and not stop:
            rounds += 1
            pending = [u for u in frontier if u not in visited]
            newly: Set[str] = set()
            # Render in bounded slices (~workers pages) and re-check the page-count
            # AND cumulative-byte budgets after EACH slice — never render the whole
            # frontier (up to MAX_PAGES) before checking, which could buffer ~20 GB
            # before the first check and OOM. Transient memory is bounded to one
            # slice's bodies (workers x per-response cap).
            for i in range(0, len(pending), workers):
                remaining = MAX_PAGES - len(visited)
                if remaining <= 0:
                    logger.warning(f"[sphinx] hit MAX_PAGES={MAX_PAGES} for {base}; stopping crawl.")
                    self.discovery_partial = True
                    stop = True
                    break
                slice_urls = [u for u in pending[i:i + workers] if u not in visited]
                if len(slice_urls) > remaining:
                    slice_urls = slice_urls[:remaining]
                    self.discovery_partial = True
                if not slice_urls:
                    continue
                visited.update(slice_urls)

                for url, title, md, links in _render_slice(slice_urls):
                    if not md:
                        continue
                    path = _url_to_path(url)
                    self._rendered[path] = (title or url, md, url, section_of.get(url, ""))
                    total_bytes += len(md.encode("utf-8"))
                    for link in links:
                        if link not in visited and link not in section_of:
                            section_of[link] = section_of.get(url, "")
                            newly.add(link)

                if total_bytes >= MAX_TOTAL_BYTES:
                    logger.warning(
                        f"[sphinx] hit MAX_TOTAL_BYTES={MAX_TOTAL_BYTES} for {base} "
                        f"({total_bytes} bytes rendered); stopping crawl."
                    )
                    self.discovery_partial = True
                    stop = True
                    break
            frontier = list(newly)

        docs: List[DiscoveredDoc] = []
        for path, (title, _md, url, section) in self._rendered.items():
            docs.append(DiscoveredDoc(
                path=path,
                title=title,
                source_url=url,
                fetch_url=url,
                section=section or None,
                summary=None,  # sphinx sites have no native summary
                product=product,
                lang=lang,
            ))
        logger.info(f"[sphinx] discovered {len(docs)} docs from {base} (BFS {rounds} rounds)")
        return docs

    def fetch(self, doc: DiscoveredDoc) -> str:
        cached = self._rendered.get(doc.path)
        if cached is not None:
            return cached[1]
        # cache miss (e.g. fetch without prior discover) — render on demand
        title, md, _ = render(http_get(doc.fetch_url), doc.fetch_url, self._base_url())
        return md or ""
