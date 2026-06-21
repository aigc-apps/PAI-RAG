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

# Sphinx-generated non-content pages / asset dirs we never ingest
EXCLUDE_BASENAMES = {"genindex.html", "search.html", "py-modindex.html", "modindex.html"}
EXCLUDE_PATH_PARTS = ("/_modules/", "/_sources/", "/_static/", "/_downloads/", "/_images/")


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
        workers = int(self.source_config.get("workers", 6))

        seeds = parse_toctree(http_get(base))
        section_of = {urljoin(base, "index.html"): ""}
        for href, section, _ in seeds:
            u = urljoin(base, href).split("#")[0]
            section_of.setdefault(u, section)

        visited: Set[str] = set()
        frontier = list(section_of)
        rounds = 0
        while frontier:
            rounds += 1
            batch = [u for u in frontier if u not in visited]
            visited.update(batch)
            newly: Set[str] = set()

            def _render_page(url):
                title, md, links = render(http_get(url), url, base)
                return url, title, md, links

            results = []
            if workers <= 1 or len(batch) <= 1:
                for u in batch:
                    try:
                        results.append(_render_page(u))
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"[sphinx] render failed {u}: {e}")
                        self.discovery_partial = True  # incomplete listing → block deletes
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(_render_page, u): u for u in batch}
                    for fu in as_completed(futs):
                        try:
                            results.append(fu.result())
                        except Exception as e:  # noqa: BLE001
                            logger.warning(f"[sphinx] render failed {futs[fu]}: {e}")
                            self.discovery_partial = True  # incomplete listing → block deletes

            for url, title, md, links in results:
                if not md:
                    continue
                path = _url_to_path(url)
                self._rendered[path] = (title or url, md, url, section_of.get(url, ""))
                for link in links:
                    if link not in visited and link not in section_of:
                        section_of[link] = section_of.get(url, "")
                        newly.add(link)
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
