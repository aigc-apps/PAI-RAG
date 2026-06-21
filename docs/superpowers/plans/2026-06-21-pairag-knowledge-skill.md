# PAI-RAG Knowledge Base Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only, zero-dependency Python CLI skill (`pairag`) that lets any CLI-capable agent search a running PAI-RAG knowledge base and fetch source files.

**Architecture:** A single self-contained `pairag.py` (stdlib only) exposes five subcommands — `kbs`, `search`, `catalog`, `grep`, `read` — over the service's HTTP API. A `Client` wraps `urllib` with friendly error mapping; command functions return rendered strings (markdown by default, `--json` for raw payloads); `main()` parses args, resolves config (flags → env → `~/.config/pairag/config.json` → defaults), and prints. A `SKILL.md` teaches agents the command vocabulary.

**Tech Stack:** Python 3 standard library only (`argparse`, `urllib`, `json`, `dataclasses`, `re`). Tests use `pytest` with the HTTP layer faked — no live server required.

---

## File Structure

```
skills/pairag-knowledge/
├── SKILL.md            # agent-facing instructions + command vocabulary (Task 10)
├── pairag.py           # the entire CLI: config, Client, resolve_kb, render_*, cmd_*, main (Tasks 1-9)
└── test_pairag.py      # pytest suite, HTTP layer faked (Tasks 1-9)
```

`pairag.py` is one focused file (~300 lines). It has no import-time side effects; `main()` runs only under `if __name__ == "__main__"`, so the test file imports it directly.

### Response shapes this CLI consumes (verified against the codebase)

- `POST /v1/retrieval` body `{query, knowledge_id}` → **flat** `{"records": [{content, score, title, url, metadata}]}`. Each `metadata` carries `doc_id`, `file_path`, `file_name`.
- `GET /v1/config/knowledgebases?size&query` → `{code, message, data: {items: [{id, name, description, ...}], total, pages, page, size}}`.
- `GET /v1/config/knowledgebases/{kb}/catalog?query&limit` → `{..., data: {results: [{doc_id, title, path, product, section, lang, source_url, score}], total}}`.
- `GET /v1/config/knowledgebases/{kb}/keyword?pattern&context&limit` → `{..., data: {results: [{doc_id, file_id, line, match, context, source_url, title}], scanned_files, scan_capped, limit_reached}}`.
- `GET /v1/config/knowledgebases/{kb}/file-content?doc_id&max_chars&offset` → `{..., data: {file_id, file_name, title, source_url, doc_id, content, content_length, offset, returned_chars, truncated, next_offset, degraded, metadata}}`.

Config endpoints wrap payloads in `{code, message, data}`; `/v1/retrieval` is flat. The `data_of()` helper handles both. `file-content` accepts `doc_id` **or** `file_id`; passing `doc_id` works for ids from any command (the server falls back to treating an unknown `doc_id` as a `file_id`).

---

### Task 1: Config resolution + error type

**Files:**
- Create: `skills/pairag-knowledge/pairag.py`
- Create: `skills/pairag-knowledge/test_pairag.py`

- [ ] **Step 1: Write the failing test**

```python
# skills/pairag-knowledge/test_pairag.py
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import pairag  # noqa: E402


class _Args:
    """Minimal stand-in for an argparse Namespace."""

    def __init__(self, **kw):
        self.base_url = kw.get("base_url")
        self.tenant = kw.get("tenant")
        self.token = kw.get("token")


def test_config_defaults_when_nothing_set():
    cfg = pairag.resolve_config(
        _Args(), env={}, config_path="/does/not/exist.json"
    )
    assert cfg.base_url == "http://localhost:8682"
    assert cfg.tenant is None
    assert cfg.default_kb is None
    assert cfg.token is None


def test_config_precedence_flag_over_env_over_file(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {"base_url": "http://file:1", "tenant": "tfile", "kb": "kbfile"}
        )
    )
    env = {
        "PAIRAG_BASE_URL": "http://env:2",
        "PAIRAG_TENANT_ID": "tenv",
        "PAIRAG_KB": "kbenv",
    }
    args = _Args(base_url="http://flag:3")
    cfg = pairag.resolve_config(args, env=env, config_path=str(cfg_file))
    assert cfg.base_url == "http://flag:3"  # flag wins
    assert cfg.tenant == "tenv"  # env beats file
    assert cfg.default_kb == "kbenv"  # env beats file


def test_pairag_error_is_exception():
    assert issubclass(pairag.PairagError, Exception)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pairag'` (file not created yet).

- [ ] **Step 3: Write minimal implementation**

```python
# skills/pairag-knowledge/pairag.py
#!/usr/bin/env python3
"""pairag — read-only CLI for a running PAI-RAG knowledge base service."""

import json
import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_BASE_URL = "http://localhost:8682"
DEFAULT_CONFIG_PATH = "~/.config/pairag/config.json"


class PairagError(Exception):
    """User-facing error; message is printed to stderr and the process exits 1."""


@dataclass
class Config:
    base_url: str
    tenant: Optional[str]
    default_kb: Optional[str]
    token: Optional[str]


def resolve_config(args, env, config_path=None):
    """Resolve settings with precedence: flags > env > config file > defaults."""
    path = os.path.expanduser(config_path or DEFAULT_CONFIG_PATH)
    file_cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            file_cfg = json.load(fh)

    def pick(flag, env_key, file_key, default=None):
        if flag is not None:
            return flag
        if env.get(env_key):
            return env[env_key]
        if file_cfg.get(file_key):
            return file_cfg[file_key]
        return default

    return Config(
        base_url=pick(
            getattr(args, "base_url", None),
            "PAIRAG_BASE_URL",
            "base_url",
            DEFAULT_BASE_URL,
        ),
        tenant=pick(
            getattr(args, "tenant", None), "PAIRAG_TENANT_ID", "tenant", None
        ),
        default_kb=pick(None, "PAIRAG_KB", "kb", None),
        token=pick(
            getattr(args, "token", None), "PAIRAG_TOKEN", "token", None
        ),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add skills/pairag-knowledge/pairag.py skills/pairag-knowledge/test_pairag.py
git commit -m "feat(skill): pairag config resolution + error type"
```

---

### Task 2: HTTP Client with friendly error mapping

**Files:**
- Modify: `skills/pairag-knowledge/pairag.py`
- Modify: `skills/pairag-knowledge/test_pairag.py`

- [ ] **Step 1: Write the failing test**

```python
# add to test_pairag.py
import io
import urllib.error


def _client(opener=None):
    return pairag.Client(
        pairag.Config("http://localhost:8682", None, None, None), opener=opener
    )


def test_request_builds_url_headers_and_parses_json():
    captured = {}

    def opener(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["tenant"] = req.headers.get(
            "X-tenant-id"
        )  # urllib title-cases header keys
        return io.BytesIO(
            json.dumps({"code": 200, "data": {"ok": True}}).encode()
        )

    c = pairag.Client(
        pairag.Config("http://host:8682", "acme", None, None), opener=opener
    )
    out = c.get("/v1/config/knowledgebases", {"size": 10, "query": None})
    assert out == {"code": 200, "data": {"ok": True}}
    assert (
        captured["url"] == "http://host:8682/v1/config/knowledgebases?size=10"
    )  # None param dropped
    assert captured["method"] == "GET"
    assert captured["tenant"] == "acme"


def test_post_sends_json_body():
    captured = {}

    def opener(req, timeout=None):
        captured["body"] = req.data
        captured["ctype"] = req.headers.get("Content-type")
        return io.BytesIO(json.dumps({"records": []}).encode())

    c = _client(opener=opener)
    c.post("/v1/retrieval", {"query": "hi", "knowledge_id": "k1"})
    assert json.loads(captured["body"]) == {
        "query": "hi",
        "knowledge_id": "k1",
    }
    assert captured["ctype"] == "application/json"


def test_connection_refused_is_friendly():
    def opener(req, timeout=None):
        raise urllib.error.URLError(
            ConnectionRefusedError("Connection refused")
        )

    c = _client(opener=opener)
    try:
        c.get("/v1/config/knowledgebases")
        assert False, "expected PairagError"
    except pairag.PairagError as e:
        assert "server running" in str(e).lower()
        assert "http://localhost:8682" in str(e)


def test_http_error_surfaces_status_and_message():
    def opener(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url,
            404,
            "Not Found",
            hdrs=None,
            fp=io.BytesIO(json.dumps({"message": "kb missing"}).encode()),
        )

    c = _client(opener=opener)
    try:
        c.get("/v1/config/knowledgebases/x")
        assert False, "expected PairagError"
    except pairag.PairagError as e:
        assert "404" in str(e)
        assert "kb missing" in str(e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v -k "request or post or connection or http_error"`
Expected: FAIL with `AttributeError: module 'pairag' has no attribute 'Client'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to pairag.py (imports near top)
import urllib.error
import urllib.parse
import urllib.request

HTTP_TIMEOUT = 30


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
            query = {
                k: v for k, v in params.items() if v is not None and v != ""
            }
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
        req = urllib.request.Request(
            url, data=data, headers=headers, method=method
        )
        try:
            resp = self._opener(req, timeout=HTTP_TIMEOUT)
            raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = (
                _extract_error_message(exc.read().decode("utf-8", "replace"))
                or exc.reason
            )
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add skills/pairag-knowledge/pairag.py skills/pairag-knowledge/test_pairag.py
git commit -m "feat(skill): pairag HTTP client with friendly error mapping"
```

---

### Task 3: Response unwrap + knowledge-base resolution

**Files:**
- Modify: `skills/pairag-knowledge/pairag.py`
- Modify: `skills/pairag-knowledge/test_pairag.py`

- [ ] **Step 1: Write the failing test**

```python
# add to test_pairag.py


def test_data_of_unwraps_config_envelope_and_passes_flat_through():
    assert pairag.data_of({"code": 200, "data": {"x": 1}}) == {"x": 1}
    assert pairag.data_of({"records": []}) == {
        "records": []
    }  # no "data" key -> as-is


def test_resolve_kb_hex_id_short_circuits_without_network():
    c = _client(opener=None)
    c._request = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should not call HTTP")
    )
    kb_id = "a" * 32
    assert pairag.resolve_kb(c, kb_id) == kb_id


def test_resolve_kb_by_name_matches_case_insensitively():
    c = _client(opener=None)
    c._request = lambda method, path, params=None, body=None: {
        "code": 200,
        "data": {
            "items": [
                {"id": "id-1", "name": "Docs"},
                {"id": "id-2", "name": "Wiki"},
            ]
        },
    }
    assert pairag.resolve_kb(c, "docs") == "id-1"


def test_resolve_kb_unknown_lists_available():
    c = _client(opener=None)
    c._request = lambda method, path, params=None, body=None: {
        "data": {"items": [{"id": "id-1", "name": "Docs"}]},
    }
    try:
        pairag.resolve_kb(c, "ghost")
        assert False, "expected PairagError"
    except pairag.PairagError as e:
        assert "ghost" in str(e)
        assert "Docs" in str(e)


def test_resolve_kb_none_errors_helpfully():
    c = _client(opener=None)
    try:
        pairag.resolve_kb(c, None)
        assert False, "expected PairagError"
    except pairag.PairagError as e:
        assert "--kb" in str(e) or "PAIRAG_KB" in str(e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v -k "data_of or resolve_kb"`
Expected: FAIL with `AttributeError: module 'pairag' has no attribute 'data_of'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to pairag.py (add `import re` to the import block)
import re

_HEX32 = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)


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
    available = (
        ", ".join((it.get("name") or it.get("id") or "?") for it in items)
        or "(none)"
    )
    raise PairagError(
        f"No knowledge base named '{kb}'. Available: {available}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add skills/pairag-knowledge/pairag.py skills/pairag-knowledge/test_pairag.py
git commit -m "feat(skill): pairag response unwrap + kb name/id resolution"
```

---

### Task 4: `kbs` command (discover knowledge bases)

**Files:**
- Modify: `skills/pairag-knowledge/pairag.py`
- Modify: `skills/pairag-knowledge/test_pairag.py`

- [ ] **Step 1: Write the failing test**

```python
# add to test_pairag.py


def _kbs_client():
    c = _client(opener=None)
    c._request = lambda method, path, params=None, body=None: {
        "data": {
            "items": [
                {"id": "id-1", "name": "Docs", "description": "Product docs"},
                {"id": "id-2", "name": "Wiki", "description": ""},
            ]
        },
    }
    return c


def test_cmd_kbs_markdown():
    out = pairag.cmd_kbs(_kbs_client(), query=None, as_json=False)
    assert "2 knowledge base(s):" in out
    assert "- Docs (id=id-1) — Product docs" in out
    assert "- Wiki (id=id-2)" in out


def test_cmd_kbs_json():
    out = pairag.cmd_kbs(_kbs_client(), query=None, as_json=True)
    assert json.loads(out)[0]["id"] == "id-1"


def test_cmd_kbs_empty():
    c = _client(opener=None)
    c._request = lambda method, path, params=None, body=None: {
        "data": {"items": []}
    }
    assert (
        pairag.cmd_kbs(c, query=None, as_json=False)
        == "No knowledge bases found."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v -k cmd_kbs`
Expected: FAIL with `AttributeError: module 'pairag' has no attribute 'cmd_kbs'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to pairag.py


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
    resp = client.get(
        "/v1/config/knowledgebases", {"size": 1000, "query": query}
    )
    items = (data_of(resp) or {}).get("items") or []
    return render_kbs(items, as_json)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v -k cmd_kbs`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add skills/pairag-knowledge/pairag.py skills/pairag-knowledge/test_pairag.py
git commit -m "feat(skill): pairag kbs command"
```

---

### Task 5: `search` command (semantic retrieval)

**Files:**
- Modify: `skills/pairag-knowledge/pairag.py`
- Modify: `skills/pairag-knowledge/test_pairag.py`

- [ ] **Step 1: Write the failing test**

```python
# add to test_pairag.py


def _search_client(records):
    c = _client(opener=None)
    sent = {}

    def fake(method, path, params=None, body=None):
        sent["method"], sent["path"], sent["body"] = method, path, body
        return {"records": records}

    c._request = fake
    c.sent = sent
    return c


def test_cmd_search_posts_to_retrieval_with_resolved_kb():
    records = [
        {
            "content": "Set vector_store.type to elasticsearch and provide the endpoint url.",
            "score": 0.87,
            "title": "Vector store",
            "url": "setup/vectordb.md",
            "metadata": {
                "doc_id": "doc-9",
                "file_path": "setup/vectordb.md",
                "file_name": "vectordb.md",
            },
        }
    ]
    c = _search_client(records)
    kb_id = "f" * 32
    out = pairag.cmd_search(
        c, kb_target=kb_id, query="vector index config", as_json=False
    )
    assert c.sent["method"] == "POST"
    assert c.sent["path"] == "/v1/retrieval"
    assert c.sent["body"] == {
        "query": "vector index config",
        "knowledge_id": kb_id,
    }
    assert '1 result(s) for "vector index config"' in out
    assert "[0.87] Vector store" in out
    assert "doc_id=doc-9" in out
    assert "elasticsearch" in out


def test_cmd_search_empty():
    c = _search_client([])
    out = pairag.cmd_search(
        c, kb_target="f" * 32, query="nothing", as_json=False
    )
    assert out == 'No results for "nothing".'


def test_cmd_search_json_returns_records():
    records = [
        {
            "content": "x",
            "score": 0.5,
            "title": "T",
            "metadata": {"doc_id": "d1"},
        }
    ]
    c = _search_client(records)
    out = pairag.cmd_search(c, kb_target="f" * 32, query="q", as_json=True)
    assert json.loads(out) == records
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v -k cmd_search`
Expected: FAIL with `AttributeError: module 'pairag' has no attribute 'cmd_search'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to pairag.py

SNIPPET_CHARS = 200


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
        tail = " · ".join(
            x for x in [f"doc_id={ident}" if ident else "", loc] if x
        )
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
    resp = client.post(
        "/v1/retrieval", {"query": query, "knowledge_id": kb_id}
    )
    records = resp.get("records") or []
    return render_search(query, kb_target or kb_id, records, as_json)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v -k cmd_search`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add skills/pairag-knowledge/pairag.py skills/pairag-knowledge/test_pairag.py
git commit -m "feat(skill): pairag search command"
```

---

### Task 6: `catalog` command (browse documents by metadata)

**Files:**
- Modify: `skills/pairag-knowledge/pairag.py`
- Modify: `skills/pairag-knowledge/test_pairag.py`

- [ ] **Step 1: Write the failing test**

```python
# add to test_pairag.py


def test_cmd_catalog_markdown_and_path():
    c = _client(opener=None)
    sent = {}

    def fake(method, path, params=None, body=None):
        sent["path"], sent["params"] = path, params
        return {
            "data": {
                "results": [
                    {
                        "doc_id": "d1",
                        "title": "Install",
                        "path": "guide/install.md",
                        "product": "rag",
                        "section": "setup",
                        "lang": "en",
                        "source_url": None,
                        "score": None,
                    },
                ],
                "total": 1,
            }
        }

    c._request = fake
    out = pairag.cmd_catalog(
        c, kb_target="f" * 32, query="install", limit=20, as_json=False
    )
    assert sent["path"] == "/v1/config/knowledgebases/" + "f" * 32 + "/catalog"
    assert sent["params"] == {"query": "install", "limit": 20}
    assert "1 document(s)" in out
    assert "- Install · doc_id=d1 · guide/install.md" in out
    assert "[rag/setup/en]" in out


def test_cmd_catalog_empty():
    c = _client(opener=None)
    c._request = lambda method, path, params=None, body=None: {
        "data": {"results": [], "total": 0}
    }
    out = pairag.cmd_catalog(
        c, kb_target="f" * 32, query=None, limit=20, as_json=False
    )
    assert out == "No documents found."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v -k cmd_catalog`
Expected: FAIL with `AttributeError: module 'pairag' has no attribute 'cmd_catalog'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to pairag.py


def _facets(item):
    parts = [item.get("product"), item.get("section"), item.get("lang")]
    parts = [p for p in parts if p]
    return f" [{'/'.join(parts)}]" if parts else ""


def render_catalog(results, as_json):
    if as_json:
        return json.dumps(results, indent=2, ensure_ascii=False)
    if not results:
        return "No documents found."
    lines = [f"{len(results)} document(s):", ""]
    for item in results:
        title = item.get("title") or "(untitled)"
        loc = item.get("path") or item.get("source_url") or ""
        head = f"- {title} · doc_id={item.get('doc_id')}"
        if loc:
            head += f" · {loc}"
        head += _facets(item)
        lines.append(head)
    return "\n".join(lines)


def cmd_catalog(client, kb_target, query, limit, as_json):
    kb_id = resolve_kb(client, kb_target)
    resp = client.get(
        f"/v1/config/knowledgebases/{kb_id}/catalog",
        {"query": query, "limit": limit},
    )
    results = (data_of(resp) or {}).get("results") or []
    return render_catalog(results, as_json)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v -k cmd_catalog`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add skills/pairag-knowledge/pairag.py skills/pairag-knowledge/test_pairag.py
git commit -m "feat(skill): pairag catalog command"
```

---

### Task 7: `grep` command (literal keyword search)

**Files:**
- Modify: `skills/pairag-knowledge/pairag.py`
- Modify: `skills/pairag-knowledge/test_pairag.py`

- [ ] **Step 1: Write the failing test**

```python
# add to test_pairag.py


def test_cmd_grep_markdown_with_counts():
    c = _client(opener=None)
    sent = {}

    def fake(method, path, params=None, body=None):
        sent["path"], sent["params"] = path, params
        return {
            "data": {
                "results": [
                    {
                        "doc_id": "d1",
                        "file_id": "f1",
                        "line": 42,
                        "match": "timeout = 600",
                        "context": "a\ntimeout = 600\nb",
                        "source_url": None,
                        "title": "config.py",
                    },
                ],
                "scanned_files": 7,
                "scan_capped": False,
                "limit_reached": False,
            }
        }

    c._request = fake
    out = pairag.cmd_grep(
        c,
        kb_target="f" * 32,
        pattern="timeout",
        context=2,
        limit=20,
        as_json=False,
    )
    assert sent["path"] == "/v1/config/knowledgebases/" + "f" * 32 + "/keyword"
    assert sent["params"] == {"pattern": "timeout", "context": 2, "limit": 20}
    assert '1 match(es) for "timeout" (scanned 7 file(s))' in out
    assert "- config.py · doc_id=d1 · line 42" in out
    assert "timeout = 600" in out


def test_cmd_grep_empty():
    c = _client(opener=None)
    c._request = lambda method, path, params=None, body=None: {
        "data": {
            "results": [],
            "scanned_files": 3,
            "scan_capped": False,
            "limit_reached": False,
        }
    }
    out = pairag.cmd_grep(
        c,
        kb_target="f" * 32,
        pattern="zzz",
        context=2,
        limit=20,
        as_json=False,
    )
    assert out == 'No matches for "zzz" (scanned 3 file(s)).'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v -k cmd_grep`
Expected: FAIL with `AttributeError: module 'pairag' has no attribute 'cmd_grep'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to pairag.py


def render_grep(pattern, payload, as_json):
    results = payload.get("results") or []
    scanned = payload.get("scanned_files", 0)
    if as_json:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    if not results:
        return f'No matches for "{pattern}" (scanned {scanned} file(s)).'
    lines = [
        f'{len(results)} match(es) for "{pattern}" (scanned {scanned} file(s))',
        "",
    ]
    for item in results:
        title = item.get("title") or "(untitled)"
        lines.append(
            f"- {title} · doc_id={item.get('doc_id')} · line {item.get('line')}"
        )
        match = (item.get("match") or "").strip()
        if match:
            lines.append(f"    {match}")
    if payload.get("scan_capped"):
        lines.append("")
        lines.append(
            "(scan capped — not all files were searched; narrow the pattern or KB)"
        )
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v -k cmd_grep`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add skills/pairag-knowledge/pairag.py skills/pairag-knowledge/test_pairag.py
git commit -m "feat(skill): pairag grep command"
```

---

### Task 8: `read` command (fetch file content)

**Files:**
- Modify: `skills/pairag-knowledge/pairag.py`
- Modify: `skills/pairag-knowledge/test_pairag.py`

- [ ] **Step 1: Write the failing test**

```python
# add to test_pairag.py


def test_cmd_read_sends_doc_id_and_renders_header_plus_content():
    c = _client(opener=None)
    sent = {}

    def fake(method, path, params=None, body=None):
        sent["path"], sent["params"] = path, params
        return {
            "data": {
                "file_id": "f1",
                "file_name": "vectordb.md",
                "title": "Vector store",
                "source_url": "https://example/vectordb.md",
                "doc_id": "d1",
                "content": "# Vector store\nSet the type.",
                "content_length": 28,
                "offset": 0,
                "returned_chars": 28,
                "truncated": False,
                "next_offset": None,
                "degraded": None,
                "metadata": {},
            }
        }

    c._request = fake
    out = pairag.cmd_read(
        c,
        kb_target="f" * 32,
        ident="d1",
        max_chars=None,
        offset=0,
        as_json=False,
    )
    assert (
        sent["path"]
        == "/v1/config/knowledgebases/" + "f" * 32 + "/file-content"
    )
    assert sent["params"] == {"doc_id": "d1", "max_chars": None, "offset": 0}
    assert "Vector store" in out
    assert "file_id=f1" in out
    assert "Set the type." in out


def test_cmd_read_truncation_note():
    c = _client(opener=None)
    c._request = lambda method, path, params=None, body=None: {
        "data": {
            "file_id": "f1",
            "file_name": "big.md",
            "title": "Big",
            "source_url": None,
            "doc_id": "d1",
            "content": "partial",
            "content_length": 100,
            "offset": 0,
            "returned_chars": 7,
            "truncated": True,
            "next_offset": 7,
            "degraded": None,
            "metadata": {},
        }
    }
    out = pairag.cmd_read(
        c, kb_target="f" * 32, ident="d1", max_chars=7, offset=0, as_json=False
    )
    assert "truncated" in out.lower()
    assert "--offset 7" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v -k cmd_read`
Expected: FAIL with `AttributeError: module 'pairag' has no attribute 'cmd_read'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to pairag.py


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
```

Note: `offset` is `0` by default; `_request` drops only `None`/`""` params, so `offset=0` is sent correctly.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v -k cmd_read`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add skills/pairag-knowledge/pairag.py skills/pairag-knowledge/test_pairag.py
git commit -m "feat(skill): pairag read command"
```

---

### Task 9: Argument parser, dispatch, and `main()`

**Files:**
- Modify: `skills/pairag-knowledge/pairag.py`
- Modify: `skills/pairag-knowledge/test_pairag.py`

- [ ] **Step 1: Write the failing test**

```python
# add to test_pairag.py
import contextlib


def _run_main(argv, fake_client):
    """Run main() with pairag.Client replaced by a factory returning fake_client."""
    original = pairag.Client
    pairag.Client = lambda config, opener=None: fake_client
    buf = io.StringIO()
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            code = pairag.main(argv)
    finally:
        pairag.Client = original
    return code, buf.getvalue(), err.getvalue()


def test_main_kbs_prints_and_exits_zero():
    fake = _kbs_client()
    code, out, err = _run_main(["kbs"], fake)
    assert code == 0
    assert "knowledge base(s)" in out


def test_main_json_flag_after_subcommand():
    fake = _kbs_client()
    code, out, _ = _run_main(["kbs", "--json"], fake)
    assert code == 0
    assert json.loads(out)[0]["id"] == "id-1"


def test_main_search_uses_default_kb_from_env(monkeypatch):
    monkeypatch.setenv("PAIRAG_KB", "f" * 32)
    fake = _search_client(
        [
            {
                "content": "hi",
                "score": 0.4,
                "title": "T",
                "metadata": {"doc_id": "d1"},
            }
        ]
    )
    code, out, _ = _run_main(["search", "q"], fake)
    assert code == 0
    assert fake.sent["body"]["knowledge_id"] == "f" * 32


def test_main_maps_pairag_error_to_exit_1():
    fake = _client(opener=None)

    def boom(method, path, params=None, body=None):
        raise pairag.PairagError("kaboom")

    fake._request = boom
    code, out, err = _run_main(["kbs"], fake)
    assert code == 1
    assert "kaboom" in err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v -k main`
Expected: FAIL with `AttributeError: module 'pairag' has no attribute 'main'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to pairag.py (add `import argparse` and `import sys` to the import block)
import argparse
import sys


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base-url", dest="base_url")
    common.add_argument("--tenant")
    common.add_argument("--token")
    common.add_argument("--kb")
    common.add_argument("--json", dest="as_json", action="store_true")

    parser = argparse.ArgumentParser(
        prog="pairag",
        description="Read-only CLI for a running PAI-RAG knowledge base.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_kbs = sub.add_parser(
        "kbs", parents=[common], help="List knowledge bases"
    )
    p_kbs.add_argument("query", nargs="?", default=None)

    p_search = sub.add_parser(
        "search", parents=[common], help="Semantic search"
    )
    p_search.add_argument("query")

    p_catalog = sub.add_parser(
        "catalog", parents=[common], help="Browse documents by metadata"
    )
    p_catalog.add_argument("--query", dest="cat_query", default=None)
    p_catalog.add_argument("--limit", type=int, default=20)

    p_grep = sub.add_parser(
        "grep", parents=[common], help="Literal keyword search"
    )
    p_grep.add_argument("pattern")
    p_grep.add_argument("--context", type=int, default=2)
    p_grep.add_argument("--limit", type=int, default=20)

    p_read = sub.add_parser(
        "read", parents=[common], help="Fetch a file's text by id"
    )
    p_read.add_argument("id")
    p_read.add_argument(
        "--max-chars", dest="max_chars", type=int, default=None
    )
    p_read.add_argument("--offset", type=int, default=0)

    return parser


def dispatch(args, config, client):
    kb_target = args.kb or config.default_kb
    if args.command == "kbs":
        return cmd_kbs(client, args.query, args.as_json)
    if args.command == "search":
        return cmd_search(client, kb_target, args.query, args.as_json)
    if args.command == "catalog":
        return cmd_catalog(
            client, kb_target, args.cat_query, args.limit, args.as_json
        )
    if args.command == "grep":
        return cmd_grep(
            client,
            kb_target,
            args.pattern,
            args.context,
            args.limit,
            args.as_json,
        )
    if args.command == "read":
        return cmd_read(
            client,
            kb_target,
            args.id,
            args.max_chars,
            args.offset,
            args.as_json,
        )
    raise PairagError(f"Unknown command: {args.command}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = resolve_config(args, os.environ)
    client = Client(config)
    try:
        output = dispatch(args, config, client)
    except PairagError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the full test suite**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v`
Expected: PASS (all tests across Tasks 1-9).

- [ ] **Step 5: Smoke-test the CLI help wiring**

Run: `python skills/pairag-knowledge/pairag.py --help` then `python skills/pairag-knowledge/pairag.py search --help`
Expected: top-level help lists `kbs, search, catalog, grep, read`; `search --help` shows `query`, `--kb`, `--json`. Exit code 0 for both.

- [ ] **Step 6: Commit**

```bash
git add skills/pairag-knowledge/pairag.py skills/pairag-knowledge/test_pairag.py
git commit -m "feat(skill): pairag argparse, dispatch, and main entrypoint"
```

---

### Task 10: SKILL.md

**Files:**
- Create: `skills/pairag-knowledge/SKILL.md`

- [ ] **Step 1: Write SKILL.md**

Create `skills/pairag-knowledge/SKILL.md` with exactly this content:

````markdown
---
name: pairag-knowledge
description: Search a PAI-RAG knowledge base and fetch source files from the command line. Use when the user asks to look something up in the knowledge base, search docs/RAG, find where something is documented, retrieve a passage with a citation, or read a knowledge-base file. Triggers — knowledge base, retrieval, RAG, "search the docs", "look it up in the KB", PAI-RAG.
---

# PAI-RAG Knowledge Base

Read-only command-line access to a running PAI-RAG service. Use it to find and
cite knowledge-base content. All commands print compact markdown; add `--json`
for raw output.

Run via: `python <skill-dir>/pairag.py <command> [...]` (Python 3, no dependencies).

## Which command

- **`search <query>`** — semantic / hybrid retrieval. Use for meaning-based
  questions ("how do I configure the vector store?"). Returns ranked passages,
  each with a `doc_id`.
- **`catalog`** — browse the document catalog by metadata (no body reads). Use to
  see what documents exist: `--query <text>` to filter, `--limit N`.
- **`grep <pattern>`** — literal keyword search over document bodies, with line
  numbers and context. Use for exact strings (an error message, a config key).
- **`read <id>`** — fetch a file's full text. Pass the `doc_id` (or `file_id`)
  from any `search` / `catalog` / `grep` result. Supports `--max-chars` and
  `--offset` for paging large files.
- **`kbs [query]`** — list available knowledge bases (id, name, description).

## Targeting a knowledge base

Every command except `kbs` needs a KB. Pass `--kb <name-or-id>`, or set a default
once with `PAIRAG_KB`. Names are resolved to ids automatically; a 32-char hex
value is treated as an id directly. If you don't know the KB, run `kbs` first.

## Configuration

Resolution order: flags → environment → `~/.config/pairag/config.json` → defaults.

| Setting   | Flag         | Env                | Default                 |
|-----------|--------------|--------------------|-------------------------|
| Base URL  | `--base-url` | `PAIRAG_BASE_URL`  | `http://localhost:8682` |
| Tenant    | `--tenant`   | `PAIRAG_TENANT_ID` | (unset)                 |
| Default KB| `--kb`       | `PAIRAG_KB`        | (unset)                 |
| Auth token| `--token`    | `PAIRAG_TOKEN`     | (unset)                 |

## Examples

```bash
# Discover knowledge bases
python pairag.py kbs

# Semantic search in the "docs" KB
python pairag.py search "how to configure the vector store" --kb docs

# Browse the catalog, then read a document by its doc_id
python pairag.py catalog --query install --kb docs
python pairag.py read d1f2... --kb docs

# Exact-string search with more context, as JSON
python pairag.py grep "timeout = 600" --kb docs --context 3 --json
```

## Citing results

Each result carries a `doc_id` (and `read` also shows `file_id`). When you relay
an answer to the user, cite the source document/file the passage came from, and
use `read` to pull the full text when you need more than the snippet.

## Notes

- The PAI-RAG server must be running. If a command reports it can't reach the
  server, confirm the service is up and `PAIRAG_BASE_URL` points at it.
- This skill is read-only: it never creates, edits, uploads, or deletes anything.
````

- [ ] **Step 2: Verify front matter and structure**

Run:
```bash
python -c "import sys; t=open('skills/pairag-knowledge/SKILL.md').read(); sys.exit(0 if t.startswith('---') and 'name: pairag-knowledge' in t and 'description:' in t else 1)"
```
Expected: exit code 0 (front matter present with `name` and `description`).

- [ ] **Step 3: Commit**

```bash
git add skills/pairag-knowledge/SKILL.md
git commit -m "docs(skill): pairag-knowledge SKILL.md"
```

---

## Final Verification

- [ ] **Run the full test suite**

Run: `pytest skills/pairag-knowledge/test_pairag.py -v`
Expected: all tests pass.

- [ ] **Confirm zero third-party imports**

Run:
```bash
python3 - <<'PY'
import ast, sys
tree = ast.parse(open('skills/pairag-knowledge/pairag.py').read())
mods = set()
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        for a in n.names:
            mods.add(a.name.split('.')[0])
    elif isinstance(n, ast.ImportFrom):
        mods.add((n.module or '').split('.')[0])
std = {'json', 'os', 're', 'sys', 'argparse', 'urllib', 'dataclasses', 'typing'}
extra = mods - std
print("non-stdlib:", sorted(extra) or "none")
sys.exit(1 if extra else 0)
PY
```
Expected: exit 0, `non-stdlib: none` — only standard-library modules are imported.

- [ ] **End-to-end smoke test against a live server (optional, if one is running)**

Run: `PAIRAG_BASE_URL=http://localhost:8682 python skills/pairag-knowledge/pairag.py kbs`
Expected: either a list of knowledge bases, or the friendly "is the server running?" message if nothing is listening — never a raw traceback.
