"""llms.txt adapter — for sites exposing an official llms.txt manifest.

Ported from docs/agent/data_source/aliyun_docs/fetch_aliyun_docs.py.

discover: fetch the llms.txt manifest and parse `- [Title](url.md): summary`
          entries grouped under `## section` headings (the manifest is
          authoritative, so add/delete detection is "free" — no body fetch).
fetch:    download the official `.md` source for one entry.

source_config:
    llms_url:  full llms.txt URL (use for sub-products), OR
    product:   product slug → https://help.aliyun.com/zh/{product}/llms.txt
    sections:  optional list of section names to include (omit = all)
    lang:      optional language tag stored on every doc (default "zh")
"""

import re
from typing import List
from urllib.parse import urlparse

from loguru import logger

from rag.datasource.base_adapter import BaseAdapter
from rag.datasource.schema import DiscoveredDoc
from rag.datasource.http_util import http_get

LLMS_URL_TEMPLATE = "https://help.aliyun.com/zh/{product}/llms.txt"
_LINE_RE = re.compile(r"-\s*\[(.+?)\]\((https?://[^)]+?\.md)\)\s*:?\s*(.*)")
_TITLE_RE = re.compile(r"^#\s+(.+)$", re.M)


def parse_llms_txt(text: str):
    """Parse an llms.txt body → [{section, title, url, summary}], grouped by `## section`."""
    items, section = [], None
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        m = _LINE_RE.match(line.strip())
        if m and section:
            items.append({
                "section": section,
                "title": m.group(1).strip(),
                "url": m.group(2).strip(),
                "summary": m.group(3).strip(),
            })
    return items


def parse_product_title(text: str) -> str:
    m = _TITLE_RE.search(text)
    return m.group(1).strip() if m else ""


def _url_to_path(url: str) -> str:
    """URL → relative path, preserving the zh/<product>/... structure."""
    return urlparse(url).path.lstrip("/")


class LlmsTxtAdapter(BaseAdapter):
    source_type = "llms_txt"
    fetched_from = "aliyun-llms.txt"

    def _llms_url(self) -> str:
        url = self.source_config.get("llms_url")
        if url:
            return url
        product = self.source_config.get("product")
        if not product:
            raise ValueError("llms_txt source_config requires either 'llms_url' or 'product'.")
        return LLMS_URL_TEMPLATE.format(product=product)

    def discover(self) -> List[DiscoveredDoc]:
        llms_url = self._llms_url()
        text = http_get(llms_url)
        product_title = parse_product_title(text) or self.source_config.get("product", "")
        items = parse_llms_txt(text)

        # Guard the common misconfiguration: a normal doc page URL instead of an
        # llms.txt manifest. Such a page parses to 0 entries; fail loudly with
        # guidance rather than silently "succeeding" with nothing ingested.
        if not items:
            hint = "" if llms_url.rstrip("/").endswith("llms.txt") else (
                " The URL does not point to an llms.txt manifest — expected a "
                "'.../llms.txt' URL (e.g. the product root + '/llms.txt')."
            )
            raise ValueError(
                f"No documents found in llms.txt at {llms_url}.{hint}"
            )

        sections = self.source_config.get("sections")
        if sections:
            wanted = set(sections)
            items = [it for it in items if it["section"] in wanted]

        lang = self.source_config.get("lang", "zh")
        docs: List[DiscoveredDoc] = []
        for it in items:
            md_url = it["url"]
            page_url = md_url[:-3] if md_url.endswith(".md") else md_url
            docs.append(DiscoveredDoc(
                path=_url_to_path(md_url),
                title=it["title"],
                source_url=page_url,
                fetch_url=md_url,
                section=it["section"],
                summary=it["summary"] or None,
                product=product_title or None,
                lang=lang,
                source_meta={"llms_summary": it["summary"]} if it["summary"] else {},
            ))
        logger.info(f"[llms_txt] discovered {len(docs)} docs from {llms_url}")
        return docs

    def fetch(self, doc: DiscoveredDoc) -> str:
        return http_get(doc.fetch_url)
