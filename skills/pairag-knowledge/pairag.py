#!/usr/bin/env python3
"""pairag — read-only CLI for a running PAI-RAG knowledge base service."""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

DEFAULT_BASE_URL = "http://localhost:8682"
HTTP_TIMEOUT = 30
SNIPPET_CHARS = 200

_HEX32 = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)


class PairagError(Exception):
    """User-facing error; message is printed to stderr and the process exits 1."""


@dataclass
class Config:
    base_url: str
    tenant: Optional[str]
    default_kb: Optional[str]
    token: Optional[str]


def resolve_config(args, env):
    """Resolve settings with precedence: flags > env > defaults."""

    def pick(flag, env_key, default=None):
        if flag is not None:
            return flag
        if env.get(env_key):
            return env[env_key]
        return default

    return Config(
        base_url=pick(getattr(args, "base_url", None), "PAIRAG_BASE_URL", DEFAULT_BASE_URL),
        tenant=pick(getattr(args, "tenant", None), "PAIRAG_TENANT_ID", None),
        default_kb=pick(None, "PAIRAG_KB", None),
        token=pick(getattr(args, "token", None), "PAIRAG_TOKEN", None),
    )


def _extract_error_message(body_txt):
    try:
        obj = json.loads(body_txt)
    except (ValueError, TypeError):
        return body_txt.strip() or None
    if isinstance(obj, dict):
        return obj.get("message") or obj.get("detail") or None
    return None


class Client:
    def __init__(self, config, opener=None):
        self.base_url = config.base_url.rstrip("/")
        self.tenant = config.tenant
        self.token = config.token
        self._opener = opener or urllib.request.urlopen

    def _request(self, method, path, params=None, body=None):
        url = self.base_url + path
        if params:
            query = {k: v for k, v in params.items() if v is not None and v != ""}
            if query:
                url += "?" + urllib.parse.urlencode(query)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.tenant:
            headers["X-TENANT-ID"] = self.tenant
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            resp = self._opener(req, timeout=HTTP_TIMEOUT)
            raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = _extract_error_message(exc.read().decode("utf-8", "replace")) or exc.reason
            raise PairagError(f"Server returned HTTP {exc.code}: {detail}")
        except urllib.error.URLError as exc:
            raise PairagError(
                f"Cannot reach PAI-RAG at {self.base_url} — is the server running? ({exc.reason})"
            )
        return json.loads(raw) if raw else {}

    def get(self, path, params=None):
        return self._request("GET", path, params=params)

    def post(self, path, body=None):
        return self._request("POST", path, body=body)


def data_of(resp):
    """Config endpoints wrap as {code,message,data}; /v1/retrieval is flat."""
    if isinstance(resp, dict) and "data" in resp:
        return resp["data"]
    return resp


def _list_kbs(client):
    resp = client.get("/v1/config/knowledgebases", {"size": 1000})
    return (data_of(resp) or {}).get("items") or []


def resolve_kb(client, kb):
    """Resolve a --kb value (name or id) to a knowledge-base id."""
    if not kb:
        raise PairagError(
            "No knowledge base specified. Pass --kb <name|id> or set PAIRAG_KB."
        )
    if _HEX32.match(kb):
        return kb
    items = _list_kbs(client)
    for item in items:
        if (item.get("name") or "").lower() == kb.lower():
            return item["id"]
    for item in items:
        if item.get("id") == kb:
            return item["id"]
    available = ", ".join((it.get("name") or it.get("id") or "?") for it in items) or "(none)"
    raise PairagError(f"No knowledge base named '{kb}'. Available: {available}")


def render_kbs(items, as_json):
    if as_json:
        return json.dumps(items, indent=2, ensure_ascii=False)
    if not items:
        return "No knowledge bases found."
    lines = [f"{len(items)} knowledge base(s):", ""]
    for item in items:
        desc = item.get("description") or ""
        line = f"- {item.get('name')} (id={item.get('id')})"
        if desc:
            line += f" — {desc}"
        lines.append(line)
    return "\n".join(lines)


def cmd_kbs(client, query, as_json):
    resp = client.get("/v1/config/knowledgebases", {"size": 1000, "query": query})
    items = (data_of(resp) or {}).get("items") or []
    return render_kbs(items, as_json)


def _snippet(text, limit=SNIPPET_CHARS):
    if not text:
        return ""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "…"


def render_search(query, label, records, as_json):
    if as_json:
        return json.dumps(records, indent=2, ensure_ascii=False)
    if not records:
        return f'No results for "{query}".'
    lines = [f'{len(records)} result(s) for "{query}" in kb={label}', ""]
    for i, rec in enumerate(records, 1):
        meta = rec.get("metadata") or {}
        ident = meta.get("doc_id") or meta.get("file_name") or ""
        loc = meta.get("file_path") or rec.get("url") or ""
        title = rec.get("title") or meta.get("file_name") or "(untitled)"
        score = rec.get("score") or 0
        head = f"{i}. [{score:.2f}] {title}"
        tail = " · ".join(x for x in [f"doc_id={ident}" if ident else "", loc] if x)
        if tail:
            head += " · " + tail
        lines.append(head)
        snippet = _snippet(rec.get("content"))
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()


def cmd_search(client, kb_target, query, as_json):
    kb_id = resolve_kb(client, kb_target)
    resp = client.post("/v1/retrieval", {"query": query, "knowledge_id": kb_id})
    records = resp.get("records") or []
    return render_search(query, kb_target or kb_id, records, as_json)


def render_catalog(items, as_json):
    if as_json:
        return json.dumps(items, indent=2, ensure_ascii=False)
    if not items:
        return "No files found."
    lines = [f"{len(items)} file(s):", ""]
    for item in items:
        meta = item.get("file_metadata") or {}
        name = item.get("file_name") or "(unnamed)"
        title = meta.get("title") or name
        source = item.get("file_source") or meta.get("source_url") or ""
        bits = []
        if name != title:
            bits.append(name)
        bits.append(f"file_id={item.get('id')}")
        if source:
            bits.append(source)
        lines.append(f"- {title} · " + " · ".join(bits))
    return "\n".join(lines)


def cmd_catalog(client, kb_target, query, limit, as_json):
    kb_id = resolve_kb(client, kb_target)
    resp = client.get(
        f"/v1/config/knowledgebases/{kb_id}/files",
        {"query": query, "size": limit},
    )
    items = (data_of(resp) or {}).get("items") or []
    return render_catalog(items, as_json)


def render_grep(pattern, payload, as_json):
    results = payload.get("results") or []
    scanned = payload.get("scanned_files", 0)
    if as_json:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    if not results:
        return f'No matches for "{pattern}" (scanned {scanned} file(s)).'
    lines = [f'{len(results)} match(es) for "{pattern}" (scanned {scanned} file(s))', ""]
    for item in results:
        title = item.get("title") or "(untitled)"
        lines.append(f"- {title} · doc_id={item.get('doc_id')} · line {item.get('line')}")
        # Render the surrounding context block (before + match + after lines) so
        # --context is reflected in the output; fall back to the match line alone.
        block = item.get("context") or item.get("match") or ""
        for ctx_line in block.splitlines():
            lines.append(f"    {ctx_line}")
    if payload.get("scan_capped"):
        lines.append("")
        lines.append("(scan capped — not all files were searched; narrow the pattern or KB)")
    if payload.get("limit_reached"):
        lines.append("")
        lines.append("(result limit reached — raise --limit for more)")
    return "\n".join(lines)


def cmd_grep(client, kb_target, pattern, context, limit, as_json):
    kb_id = resolve_kb(client, kb_target)
    resp = client.get(
        f"/v1/config/knowledgebases/{kb_id}/keyword",
        {"pattern": pattern, "context": context, "limit": limit},
    )
    return render_grep(pattern, data_of(resp) or {}, as_json)


def render_read(doc, as_json):
    if as_json:
        return json.dumps(doc, indent=2, ensure_ascii=False)
    title = doc.get("title") or doc.get("file_name") or "(untitled)"
    header = [
        title,
        " · ".join(
            x
            for x in [
                f"file_id={doc.get('file_id')}" if doc.get("file_id") else "",
                f"doc_id={doc.get('doc_id')}" if doc.get("doc_id") else "",
                doc.get("source_url") or "",
            ]
            if x
        ),
        "",
    ]
    body = doc.get("content") or "(empty)"
    out = "\n".join(header) + body
    if doc.get("truncated"):
        nxt = doc.get("next_offset")
        out += f"\n\n[truncated at {doc.get('returned_chars')} of {doc.get('content_length')} chars"
        if nxt is not None:
            out += f"; continue with --offset {nxt}"
        out += "]"
    return out


def cmd_read(client, kb_target, ident, max_chars, offset, as_json):
    kb_id = resolve_kb(client, kb_target)
    resp = client.get(
        f"/v1/config/knowledgebases/{kb_id}/file-content",
        {"doc_id": ident, "max_chars": max_chars, "offset": offset},
    )
    return render_read(data_of(resp) or {}, as_json)


def _add_common(parser, suppress):
    """Register the global options on a parser.

    They are added both to the top-level parser (with real defaults) and to each
    subparser (with ``SUPPRESS`` defaults). SUPPRESS means a subparser leaves the
    attribute untouched when the option is absent, so a value given *before* the
    subcommand (e.g. ``pairag --kb docs search q``) is not clobbered by the
    subparser's default. Either position works.
    """
    default = argparse.SUPPRESS if suppress else None
    bool_default = argparse.SUPPRESS if suppress else False
    parser.add_argument("--base-url", dest="base_url", default=default)
    parser.add_argument("--tenant", default=default)
    parser.add_argument("--token", default=default)
    parser.add_argument("--kb", default=default)
    parser.add_argument("--json", dest="as_json", action="store_true", default=bool_default)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pairag", description="Read-only CLI for a running PAI-RAG knowledge base."
    )
    _add_common(parser, suppress=False)
    sub = parser.add_subparsers(dest="command", required=True)

    p_kbs = sub.add_parser("kbs", help="List knowledge bases")
    _add_common(p_kbs, suppress=True)
    p_kbs.add_argument("query", nargs="?", default=None)

    p_search = sub.add_parser("search", help="Semantic search")
    _add_common(p_search, suppress=True)
    p_search.add_argument("query")

    p_catalog = sub.add_parser("catalog", help="Browse documents by metadata")
    _add_common(p_catalog, suppress=True)
    p_catalog.add_argument("--query", dest="cat_query", default=None)
    p_catalog.add_argument("--limit", type=int, default=20)

    p_grep = sub.add_parser("grep", help="Literal keyword search")
    _add_common(p_grep, suppress=True)
    p_grep.add_argument("pattern")
    p_grep.add_argument("--context", type=int, default=2)
    p_grep.add_argument("--limit", type=int, default=20)

    p_read = sub.add_parser("read", help="Fetch a file's text by id")
    _add_common(p_read, suppress=True)
    p_read.add_argument("id")
    p_read.add_argument("--max-chars", dest="max_chars", type=int, default=None)
    p_read.add_argument("--offset", type=int, default=0)

    return parser


def dispatch(args, config, client):
    kb_target = args.kb or config.default_kb
    if args.command == "kbs":
        return cmd_kbs(client, args.query, args.as_json)
    if args.command == "search":
        return cmd_search(client, kb_target, args.query, args.as_json)
    if args.command == "catalog":
        return cmd_catalog(client, kb_target, args.cat_query, args.limit, args.as_json)
    if args.command == "grep":
        return cmd_grep(client, kb_target, args.pattern, args.context, args.limit, args.as_json)
    if args.command == "read":
        return cmd_read(client, kb_target, args.id, args.max_chars, args.offset, args.as_json)
    raise PairagError(f"Unknown command: {args.command}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        config = resolve_config(args, os.environ)
        client = Client(config)
        output = dispatch(args, config, client)
    except PairagError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
