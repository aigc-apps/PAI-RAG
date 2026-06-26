# Agent Tools & Extensibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the agent a tight, high-quality default tool set (`current_datetime`, `web_fetch`, `web_search`) behind a `ToolRegistry`, with uniform extension points for local **skills** and **MCP** servers, wired into `build_context` and governed by the soul's `tools_enabled`.

**Architecture:** A `ToolRegistry` holds named `Tool`s and builds a `ToolBox`. Built-in tools live in `agent/tools/builtin/` as factory functions; `build_default_registry(settings)` assembles them (web_search only when a provider is configured). Skills (`load_skills`) and MCP (`register_mcp_tools`) register `Tool`s into the same registry. `build_context` selects tools (soul.tools_enabled or all), builds the `ToolBox`, and renders the system prompt naming exactly those tools. The agent's tool loop is already implemented and unchanged.

**Tech Stack:** Python 3.11, pydantic v2, `httpx` 0.28 (already a dep), `loguru`, pytest. Tests run from `backend/`: `cd backend && python -m pytest ../tests/app -q`. Async tests use `asyncio.run(run())`.

**Reference spec:** `docs/superpowers/specs/2026-06-26-agent-soul-and-tools-design.md`. **Prerequisite plan:** `2026-06-26-agent-soul.md` (the `Soul`, `render_system_prompt`, and the `build_context(..., *, soul=DEFAULT_SOUL)` signature) must be implemented first. Branch: `personal/yfei/agent-core`.

## Global Constraints

- **Import-lean:** new modules import only stdlib + `httpx` + `loguru` + `pydantic` + existing `agent.*`. Boot/import-isolation tests stay green.
- **Reuse the existing `Tool`/`ToolBox`** (`agent/tools/base.py`) verbatim — `Tool(name, description, parameters, fn, return_direct=False)`, `fn` is `async (**args) -> str`. Do not change them.
- **Tools never raise to the loop:** every tool `fn` catches its own errors and returns a concise `"<tool> failed: …"` string.
- **Web tools degrade cleanly:** `web_search` is registered only when a search provider is configured; an unconfigured search is simply absent, not a runtime error.
- **Route-compatible signature:** `build_context` gains a keyword-only `registry: Optional[ToolRegistry] = None`; with `None` it behaves exactly as after Plan A (empty `ToolBox`).
- **MCP & search are interfaces this iteration:** implement the adapter (MCP schema→`Tool`, calling an injected client) and the `SearchProvider` protocol with a configurable/fake provider — NOT a live transport or vendor. (Documented in the spec's out-of-scope.)
- Run the full app suite at the end of every task: `cd backend && python -m pytest ../tests/app -q`.

---

## Key existing contracts (verified, do not re-derive)

- **`agent/tools/base.py`** — `@dataclass Tool(name: str, description: str, parameters: dict, fn: Callable[..., Awaitable[str]], return_direct: bool = False)` with `.openai_schema()`. `ToolBox(tools: List[Tool])` with `.tools`, `.get(name)`, `.openai_schema()`, `async .dispatch(tc: ToolCall) -> ToolResult`. `ToolBox.dispatch` parses `tc.arguments` (JSON) and calls `fn(**args)` with retry, wrapping errors into a `ToolResult` (so a tool returning an error string vs raising both work).
- **`agent/message.py`** — `ToolCall(id, name, arguments)` (arguments is a JSON string).
- **`agent/agent.py`** — already advertises `ctx.tools.openai_schema()` to the LLM and dispatches via `ctx.tools.dispatch(tc)`. No change needed.
- **`agent/soul.py`** (Plan A) — `Soul.tools_enabled: Optional[List[str]]` (None = all); `render_system_prompt(soul, *, tool_names, extra="")`; `DEFAULT_SOUL`.
- **`app/builder.py`** (after Plan A) — `build_context(request, store, *, soul: Soul = DEFAULT_SOUL)`; composes `effective_soul`, currently `tool_names: List[str] = []`, `tools=ToolBox([])`. This plan adds the `registry` kwarg + real tool selection.
- **`app/deps.py`** (after Plan A) — `AppState(store, llm, default_model, context_window, max_output_tokens, soul)`.
- **`app/config.py`** (after Plan A) — `Settings` with `agent_name`, `agent_role`. This plan adds search/skills settings.
- **`app/lean_main.py`** (after Plan A) — builds `AppState(..., soul=soul)`.
- **`utils/time_utils.py`** — `get_current_time_str() -> str`.
- **`httpx`** 0.28 — `httpx.AsyncClient(timeout=, follow_redirects=True)` as an async context manager; `resp.raise_for_status()`, `resp.text`.
- **Test scaffolding** — `import sys, os[, asyncio]; sys.path.insert(0, ".../backend")`. Async via `asyncio.run(run())`.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/agent/tools/registry.py` (new) | `ToolRegistry` (register/get/names/build_toolbox) |
| `backend/agent/tools/builtin/__init__.py` (new) | package marker |
| `backend/agent/tools/builtin/datetime_tool.py` (new) | `make_current_datetime_tool()` |
| `backend/agent/tools/builtin/web_fetch.py` (new) | `make_web_fetch_tool()` + HTML→text |
| `backend/agent/tools/builtin/web_search.py` (new) | `SearchProvider` protocol + `make_web_search_tool(provider)` |
| `backend/agent/tools/defaults.py` (new) | `build_default_registry(settings, *, search_provider=None)` |
| `backend/agent/tools/skills.py` (new) | `load_skills(path, registry)` |
| `backend/agent/tools/mcp.py` (new) | `mcp_tool_to_tool` + `register_mcp_tools` |
| `backend/app/config.py` (modify) | search/skills settings |
| `backend/app/builder.py` (modify) | `registry` kwarg + tool selection |
| `backend/app/deps.py` (modify) | `AppState.registry` |
| `backend/app/routes/responses.py` (modify) | pass `registry=state.registry` |
| `backend/app/lean_main.py` (modify) | build registry (defaults + skills) |
| `tests/app/test_tool_registry.py` (new) | registry |
| `tests/app/test_builtin_tools.py` (new) | builtin tools + defaults |
| `tests/app/test_tool_extensions.py` (new) | skills + MCP |
| `tests/app/test_builder_tools.py` (new) | build_context tool wiring + dispatch |

---

## Task 1: `ToolRegistry`

**Files:**
- Create: `backend/agent/tools/registry.py`
- Test: `tests/app/test_tool_registry.py`

**Interfaces:**
- Consumes: `Tool`, `ToolBox` (`agent/tools/base.py`).
- Produces: `class ToolRegistry` with `register(tool)`, `get(name) -> Optional[Tool]`, `names() -> List[str]`, `build_toolbox(names: Optional[List[str]] = None) -> ToolBox` (None = all; unknown names skipped with a warning).

- [ ] **Step 1: Write the failing test** — `tests/app/test_tool_registry.py`

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.tools.registry import ToolRegistry
from agent.tools.base import Tool


def _tool(name):
    async def fn():
        return name
    return Tool(name=name, description=name, parameters={"type": "object", "properties": {}}, fn=fn)


def test_register_get_names():
    r = ToolRegistry()
    r.register(_tool("a"))
    r.register(_tool("b"))
    assert r.get("a").name == "a"
    assert r.get("missing") is None
    assert set(r.names()) == {"a", "b"}


def test_build_toolbox_all_and_subset_and_unknown():
    r = ToolRegistry()
    r.register(_tool("a"))
    r.register(_tool("b"))
    assert {t.name for t in r.build_toolbox().tools} == {"a", "b"}
    assert [t.name for t in r.build_toolbox(["b"]).tools] == ["b"]
    # unknown names are skipped, known ones kept
    assert [t.name for t in r.build_toolbox(["b", "nope"]).tools] == ["b"]


def test_reregister_overwrites():
    r = ToolRegistry()
    r.register(_tool("a"))
    new = _tool("a")
    r.register(new)
    assert r.get("a") is new
    assert r.names().count("a") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_tool_registry.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.tools.registry'`.

- [ ] **Step 3: Implement `backend/agent/tools/registry.py`**

```python
from __future__ import annotations
from typing import Dict, List, Optional
from loguru import logger
from agent.tools.base import Tool, ToolBox


class ToolRegistry:
    """Named collection of Tools; builds the ToolBox the agent runs with.
    Skills and MCP servers register into the same registry, so tool selection
    (soul.tools_enabled) is uniform regardless of where a tool came from."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.warning(f"tool '{tool.name}' re-registered; overwriting")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def build_toolbox(self, names: Optional[List[str]] = None) -> ToolBox:
        if names is None:
            return ToolBox(list(self._tools.values()))
        out: List[Tool] = []
        for n in names:
            tool = self._tools.get(n)
            if tool is None:
                logger.warning(f"tool '{n}' requested but not registered; skipping")
            else:
                out.append(tool)
        return ToolBox(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_tool_registry.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/agent/tools/registry.py tests/app/test_tool_registry.py
git commit -m "feat(tools): ToolRegistry (register/get/names/build_toolbox)"
```

---

## Task 2: Built-in tools + `build_default_registry`

The default trio + the assembler. `web_search` is provider-injected and only registered when configured.

**Files:**
- Create: `backend/agent/tools/builtin/__init__.py`, `datetime_tool.py`, `web_fetch.py`, `web_search.py`, `backend/agent/tools/defaults.py`
- Modify: `backend/app/config.py`
- Test: `tests/app/test_builtin_tools.py`

**Interfaces:**
- Consumes: `Tool` (base), `ToolRegistry` (Task 1), `httpx`, `utils.time_utils`.
- Produces:
  - `make_current_datetime_tool() -> Tool` (name `current_datetime`, no args).
  - `make_web_fetch_tool(client_factory=None, limit=8000) -> Tool` (name `web_fetch`, arg `url`); `client_factory()` returns an httpx-like async-context client (default: real `httpx.AsyncClient`).
  - `SearchProvider` (Protocol: `async def search(query: str, num_results: int) -> List[dict]`); `make_web_search_tool(provider) -> Tool` (name `web_search`, args `query`, `num_results`).
  - `build_default_registry(settings, *, search_provider: Optional[SearchProvider] = None) -> ToolRegistry` — always `current_datetime` + `web_fetch`; adds `web_search` when `search_provider` is given or `settings.search_provider != "none"`.
  - `Settings.search_provider="none"`, `search_api_key=""`, `search_endpoint=""`, `skills_dir=""`.

- [ ] **Step 1: Write the failing test** — `tests/app/test_builtin_tools.py`

```python
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.tools.builtin.datetime_tool import make_current_datetime_tool
from agent.tools.builtin.web_fetch import make_web_fetch_tool
from agent.tools.builtin.web_search import make_web_search_tool
from agent.tools.defaults import build_default_registry


def test_current_datetime_tool_returns_a_time_string():
    t = make_current_datetime_tool()
    assert t.name == "current_datetime"
    out = asyncio.run(t.fn())
    assert isinstance(out, str) and len(out) >= 8


class _FakeResp:
    def __init__(self, text):
        self.text = text
    def raise_for_status(self):
        return None


class _FakeClient:
    def __init__(self, resp, boom=False):
        self._resp = resp
        self._boom = boom
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        return False
    async def get(self, url):
        if self._boom:
            raise RuntimeError("network down")
        return self._resp


def test_web_fetch_extracts_text_and_truncates():
    html = "<html><head><style>x{}</style></head><body><h1>Hi</h1><p>World &amp; more</p><script>bad()</script></body></html>"
    t = make_web_fetch_tool(client_factory=lambda: _FakeClient(_FakeResp(html)), limit=50)
    out = asyncio.run(t.fn(url="http://x"))
    assert "Hi" in out and "World" in out
    assert "bad()" not in out and "<" not in out
    assert len(out) <= 50


def test_web_fetch_returns_error_string_on_failure():
    t = make_web_fetch_tool(client_factory=lambda: _FakeClient(None, boom=True))
    out = asyncio.run(t.fn(url="http://x"))
    assert out.startswith("web_fetch failed:")


class _FakeProvider:
    async def search(self, query, num_results):
        return [{"title": "T1", "url": "http://1", "snippet": "S1"},
                {"title": "T2", "url": "http://2", "snippet": "S2"}][:num_results]


def test_web_search_formats_results():
    t = make_web_search_tool(_FakeProvider())
    out = asyncio.run(t.fn(query="hello", num_results=2))
    assert "T1" in out and "http://2" in out and "S2" in out


def test_web_search_returns_error_string_on_provider_failure():
    class _Boom:
        async def search(self, q, n):
            raise RuntimeError("boom")
    t = make_web_search_tool(_Boom())
    out = asyncio.run(t.fn(query="x"))
    assert out.startswith("web_search failed:")


class _Settings:
    search_provider = "none"


def test_default_registry_omits_search_when_unconfigured():
    reg = build_default_registry(_Settings())
    assert set(reg.names()) == {"current_datetime", "web_fetch"}


def test_default_registry_includes_injected_search_provider():
    reg = build_default_registry(_Settings(), search_provider=_FakeProvider())
    assert "web_search" in reg.names()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_builtin_tools.py -q`
Expected: FAIL — modules don't exist.

- [ ] **Step 3: Create `backend/agent/tools/builtin/__init__.py`** (empty package marker)

```python
```

- [ ] **Step 4: Implement `backend/agent/tools/builtin/datetime_tool.py`**

```python
from __future__ import annotations
from agent.tools.base import Tool
from utils.time_utils import get_current_time_str


def make_current_datetime_tool() -> Tool:
    async def fn() -> str:
        return get_current_time_str()

    return Tool(
        name="current_datetime",
        description="Return the current local date and time (use for 'what day/time is it' and time-relative reasoning).",
        parameters={"type": "object", "properties": {}, "required": []},
        fn=fn,
    )
```

- [ ] **Step 5: Implement `backend/agent/tools/builtin/web_fetch.py`**

```python
from __future__ import annotations
import re
from typing import Callable, Optional
import httpx
from agent.tools.base import Tool

_DROP_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITIES = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&nbsp;": " "}


def _html_to_text(html: str, limit: int) -> str:
    html = _DROP_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", html)
    for ent, ch in _ENTITIES.items():
        text = text.replace(ent, ch)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def make_web_fetch_tool(
    client_factory: Optional[Callable[[], object]] = None, limit: int = 8000
) -> Tool:
    """`client_factory()` returns an async-context HTTP client with `.get(url)`
    (default: a real httpx.AsyncClient). Injected as a fake in tests."""

    async def fn(url: str) -> str:
        try:
            client_cm = (
                client_factory()
                if client_factory is not None
                else httpx.AsyncClient(timeout=15, follow_redirects=True)
            )
            async with client_cm as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return _html_to_text(resp.text, limit)
        except Exception as ex:  # never raise into the agent loop
            return f"web_fetch failed: {ex}"

    return Tool(
        name="web_fetch",
        description="Fetch a URL and return its readable text content.",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to fetch."}},
            "required": ["url"],
        },
        fn=fn,
    )
```

- [ ] **Step 6: Implement `backend/agent/tools/builtin/web_search.py`**

```python
from __future__ import annotations
from typing import Dict, List, Protocol
from agent.tools.base import Tool


class SearchProvider(Protocol):
    async def search(self, query: str, num_results: int) -> List[Dict]: ...


def _format(results: List[Dict]) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r.get('title', '(no title)')}\n   {r.get('url', '')}\n   {r.get('snippet', '')}"
        )
    return "\n".join(lines)


def make_web_search_tool(provider: SearchProvider) -> Tool:
    async def fn(query: str, num_results: int = 5) -> str:
        try:
            results = await provider.search(query, num_results)
            return _format(results)
        except Exception as ex:
            return f"web_search failed: {ex}"

    return Tool(
        name="web_search",
        description="Search the web for current information. Returns ranked results with titles, URLs, and snippets.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "num_results": {"type": "integer", "description": "How many results to return (default 5)."},
            },
            "required": ["query"],
        },
        fn=fn,
    )
```

- [ ] **Step 7: Implement `backend/agent/tools/defaults.py`**

```python
from __future__ import annotations
from typing import Optional
from loguru import logger
from agent.tools.registry import ToolRegistry
from agent.tools.builtin.datetime_tool import make_current_datetime_tool
from agent.tools.builtin.web_fetch import make_web_fetch_tool
from agent.tools.builtin.web_search import make_web_search_tool, SearchProvider


def build_default_registry(
    settings, *, search_provider: Optional[SearchProvider] = None
) -> ToolRegistry:
    """Assemble the default registry. current_datetime + web_fetch always; web_search
    only when a provider is injected or `settings.search_provider != "none"`."""
    reg = ToolRegistry()
    reg.register(make_current_datetime_tool())
    reg.register(make_web_fetch_tool())

    provider = search_provider
    if provider is None and getattr(settings, "search_provider", "none") != "none":
        provider = _provider_from_settings(settings)
    if provider is not None:
        reg.register(make_web_search_tool(provider))
    return reg


def _provider_from_settings(settings) -> Optional[SearchProvider]:
    """Hook for a real search backend (Tavily/Serp/etc.). Not wired to a vendor
    this iteration — returns None so web_search stays off unless a provider is
    injected explicitly. See the design's out-of-scope."""
    logger.info(
        f"search_provider={getattr(settings, 'search_provider', 'none')} configured "
        "but no live provider is wired yet; web_search disabled."
    )
    return None
```

- [ ] **Step 8: Add settings** in `backend/app/config.py`

Add to `Settings` (after `agent_role`):

```python
    search_provider: str = "none"
    search_api_key: str = ""
    search_endpoint: str = ""
    skills_dir: str = ""
```

- [ ] **Step 9: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_builtin_tools.py -q`
Expected: PASS (7 tests).

- [ ] **Step 10: Commit**

```bash
git add backend/agent/tools/builtin backend/agent/tools/defaults.py backend/app/config.py tests/app/test_builtin_tools.py
git commit -m "feat(tools): builtin tool trio (current_datetime/web_fetch/web_search) + build_default_registry"
```

---

## Task 3: Extensibility — skills loader + MCP adapter

The two extension paths, both feeding the same registry. Pure/local and fully tested; live transports are deferred (see spec).

**Files:**
- Create: `backend/agent/tools/skills.py`, `backend/agent/tools/mcp.py`
- Test: `tests/app/test_tool_extensions.py`

**Interfaces:**
- Consumes: `Tool` (base), `ToolRegistry` (Task 1).
- Produces:
  - `load_skills(path: str, registry: ToolRegistry) -> List[str]` — import each `*.py` (not starting with `_`) under `path` exposing `get_tools() -> List[Tool]`; register their tools; return registered names; a broken skill is logged and skipped.
  - `mcp_tool_to_tool(spec: dict, call: CallFn, prefix: str = "") -> Tool` — map an MCP tool descriptor (`{name, description, inputSchema}`) to a `Tool` whose `fn(**kwargs)` awaits `call(remote_name, kwargs)` and stringifies the result. `CallFn = Callable[[str, dict], Awaitable[object]]`.
  - `register_mcp_tools(specs: List[dict], call: CallFn, registry: ToolRegistry, prefix: str = "") -> List[str]`.

- [ ] **Step 1: Write the failing test** — `tests/app/test_tool_extensions.py`

```python
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.tools.registry import ToolRegistry
from agent.tools.skills import load_skills
from agent.tools.mcp import mcp_tool_to_tool, register_mcp_tools


_SKILL_SRC = '''
from agent.tools.base import Tool

async def _hello(name: str = "world"):
    return f"hello {name}"

def get_tools():
    return [Tool(name="hello", description="greet",
                 parameters={"type": "object", "properties": {"name": {"type": "string"}}},
                 fn=_hello)]
'''

_BROKEN_SRC = "this is not valid python ("


def test_load_skills_registers_tools(tmp_path):
    (tmp_path / "greet.py").write_text(_SKILL_SRC)
    (tmp_path / "broken.py").write_text(_BROKEN_SRC)
    (tmp_path / "_ignored.py").write_text("raise RuntimeError('should not load')")
    reg = ToolRegistry()
    names = load_skills(str(tmp_path), reg)
    assert "hello" in names
    assert reg.get("hello") is not None
    assert asyncio.run(reg.get("hello").fn(name="x")) == "hello x"


def test_load_skills_missing_dir_is_noop():
    reg = ToolRegistry()
    assert load_skills("/no/such/dir", reg) == []


def test_mcp_tool_to_tool_maps_schema_and_calls_client():
    calls = []

    async def call(name, args):
        calls.append((name, args))
        return {"ok": True, "echo": args}

    spec = {"name": "lookup", "description": "look up", "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}}
    tool = mcp_tool_to_tool(spec, call)
    assert tool.name == "lookup"
    assert tool.parameters["properties"]["q"]["type"] == "string"
    out = asyncio.run(tool.fn(q="hi"))
    assert calls == [("lookup", {"q": "hi"})]
    assert '"echo"' in out and "hi" in out  # dict result stringified as JSON


def test_register_mcp_tools_namespaces_with_prefix():
    async def call(name, args):
        return "ok"

    reg = ToolRegistry()
    specs = [{"name": "a", "description": "", "inputSchema": {"type": "object", "properties": {}}},
             {"name": "b", "description": "", "inputSchema": {"type": "object", "properties": {}}}]
    names = register_mcp_tools(specs, call, reg, prefix="srv.")
    assert names == ["srv.a", "srv.b"]
    assert reg.get("srv.a") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_tool_extensions.py -q`
Expected: FAIL — modules don't exist.

- [ ] **Step 3: Implement `backend/agent/tools/skills.py`**

```python
from __future__ import annotations
import importlib.util
from pathlib import Path
from typing import List
from loguru import logger
from agent.tools.registry import ToolRegistry


def load_skills(path: str, registry: ToolRegistry) -> List[str]:
    """Import each *.py skill under `path` exposing `get_tools() -> list[Tool]` and
    register its tools. Returns registered tool names. A skill that fails to import
    is logged and skipped (one bad skill never breaks boot). Files starting with
    '_' are ignored."""
    registered: List[str] = []
    root = Path(path)
    if not root.exists() or not root.is_dir():
        return registered
    for py in sorted(root.glob("*.py")):
        if py.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"skill_{py.stem}", py)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            get_tools = getattr(mod, "get_tools", None)
            if get_tools is None:
                logger.warning(f"skill {py.name} has no get_tools(); skipping")
                continue
            for tool in get_tools():
                registry.register(tool)
                registered.append(tool.name)
        except Exception:
            logger.exception(f"failed to load skill {py}")
    return registered
```

- [ ] **Step 4: Implement `backend/agent/tools/mcp.py`**

```python
from __future__ import annotations
import json
from typing import Awaitable, Callable, List
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry

# Injected by a live MCP client: call(remote_tool_name, args) -> result (awaitable).
CallFn = Callable[[str, dict], Awaitable[object]]


def mcp_tool_to_tool(spec: dict, call: CallFn, prefix: str = "") -> Tool:
    """Map an MCP tool descriptor ({name, description, inputSchema}) to a Tool.
    The Tool's fn forwards parsed kwargs to the injected MCP `call` and stringifies
    the result. `prefix` namespaces the local name to avoid collisions."""
    remote_name = spec["name"]
    local_name = f"{prefix}{remote_name}" if prefix else remote_name
    parameters = spec.get("inputSchema") or {"type": "object", "properties": {}}

    async def fn(**kwargs) -> str:
        result = await call(remote_name, kwargs)
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False)
        except Exception:
            return str(result)

    return Tool(
        name=local_name,
        description=spec.get("description", ""),
        parameters=parameters,
        fn=fn,
    )


def register_mcp_tools(
    specs: List[dict], call: CallFn, registry: ToolRegistry, prefix: str = ""
) -> List[str]:
    """Register all tools a (single) MCP server exposes, returning the local names."""
    names: List[str] = []
    for spec in specs:
        tool = mcp_tool_to_tool(spec, call, prefix=prefix)
        registry.register(tool)
        names.append(tool.name)
    return names
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_tool_extensions.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add backend/agent/tools/skills.py backend/agent/tools/mcp.py tests/app/test_tool_extensions.py
git commit -m "feat(tools): skill loader + MCP tool adapter (extension points into the registry)"
```

---

## Task 4: Wire tools into `build_context` + app

Select tools from the registry (governed by `soul.tools_enabled`), build the `ToolBox`, render the prompt naming exactly those tools, and assemble the default registry at boot.

**Files:**
- Modify: `backend/app/builder.py`, `backend/app/deps.py`, `backend/app/routes/responses.py`, `backend/app/lean_main.py`
- Test: `tests/app/test_builder_tools.py`

**Interfaces:**
- Consumes: `ToolRegistry` (Task 1), `build_default_registry` (Task 2), `load_skills` (Task 3); `render_system_prompt`/`Soul` (Plan A).
- Produces:
  - `build_context(request, store, *, soul: Soul = DEFAULT_SOUL, registry: Optional[ToolRegistry] = None)` — selects `tool_names = effective_soul.tools_enabled or registry.names()`, builds the `ToolBox`, renders the prompt with the box's actual tool names. `registry=None` → empty `ToolBox` (Plan-A behavior).
  - `AppState.registry: ToolRegistry` (default empty).

- [ ] **Step 1: Write the failing test** — `tests/app/test_builder_tools.py`

```python
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.schemas import ResponsesRequest
from app.builder import build_context
from app.store.memory import InMemoryStore
from agent.tools.defaults import build_default_registry
from agent.message import ToolCall


class _Settings:
    search_provider = "none"


def test_build_context_wires_registry_tools_and_names_them_in_prompt():
    async def run():
        reg = build_default_registry(_Settings())
        ctx, _ = await build_context(
            ResponsesRequest(model="m", input="hi"), InMemoryStore(), registry=reg
        )
        names = {t.name for t in ctx.tools.tools}
        assert names == {"current_datetime", "web_fetch"}
        assert "current_datetime" in ctx.system_prompt
        assert "web_fetch" in ctx.system_prompt

    asyncio.run(run())


def test_soul_tools_enabled_filters_the_toolbox():
    async def run():
        reg = build_default_registry(_Settings())
        req = ResponsesRequest(
            model="m", input="hi", soul={"tools_enabled": ["current_datetime"]}
        )
        ctx, _ = await build_context(req, InMemoryStore(), registry=reg)
        assert [t.name for t in ctx.tools.tools] == ["current_datetime"]
        assert "web_fetch" not in ctx.system_prompt

    asyncio.run(run())


def test_no_registry_means_no_tools():
    async def run():
        ctx, _ = await build_context(
            ResponsesRequest(model="m", input="hi"), InMemoryStore()
        )
        assert ctx.tools.tools == []
        assert "no tools" in ctx.system_prompt.lower()

    asyncio.run(run())


def test_wired_tool_is_dispatchable():
    async def run():
        reg = build_default_registry(_Settings())
        ctx, _ = await build_context(
            ResponsesRequest(model="m", input="hi"), InMemoryStore(), registry=reg
        )
        result = await ctx.tools.dispatch(
            ToolCall(id="c1", name="current_datetime", arguments="{}")
        )
        assert result.ok and isinstance(result.content, str) and len(result.content) >= 8

    asyncio.run(run())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_builder_tools.py -q`
Expected: FAIL — `build_context` has no `registry` kwarg / tools not wired.

- [ ] **Step 3: Wire the registry in `backend/app/builder.py`**

Add the import:

```python
from agent.tools.registry import ToolRegistry
```

Replace the `build_context` signature + the tool-selection / prompt-render section (the part after `effective_soul = soul.merge(override)`):

```python
async def build_context(
    request: ResponsesRequest,
    store,
    *,
    soul: Soul = DEFAULT_SOUL,
    registry: Optional[ToolRegistry] = None,
) -> Tuple[AgentContext, Optional[str]]:
```

and replace the tool/prompt block with:

```python
    override = dict(request.soul or {})
    if request.instructions:
        override["extra_instructions"] = request.instructions
    effective_soul = soul.merge(override)

    if registry is not None:
        selected = (
            effective_soul.tools_enabled
            if effective_soul.tools_enabled is not None
            else registry.names()
        )
        toolbox = registry.build_toolbox(selected)
    else:
        toolbox = ToolBox([])

    tool_names = [t.name for t in toolbox.tools]
    system_prompt = render_system_prompt(effective_soul, tool_names=tool_names)

    ctx = AgentContext(
        system_prompt=system_prompt,
        history=items_to_messages(history_items),
        current_turn=_input_to_turn(request.input),
        attachments=[],
        hints=[],
        tools=toolbox,
        run_vars=RunVars(),
    )
    return ctx, conversation_id
```

(`Optional` is already imported in `builder.py`.)

- [ ] **Step 4: Add `AppState.registry`** in `backend/app/deps.py`

Add the import:

```python
from agent.tools.registry import ToolRegistry
```

Add the field (after `soul`):

```python
    registry: ToolRegistry = field(default_factory=ToolRegistry)
```

- [ ] **Step 5: Pass the registry through the route** in `backend/app/routes/responses.py`

```python
        ctx, conversation_id = await build_context(
            request, state.store, soul=state.soul, registry=state.registry
        )
```

- [ ] **Step 6: Build the default registry at boot** in `backend/app/lean_main.py`

Add imports:

```python
from agent.tools.defaults import build_default_registry
from agent.tools.skills import load_skills
```

In `lifespan`, after building `soul`, build the registry and pass it to `AppState`:

```python
    registry = build_default_registry(settings)
    if settings.skills_dir:
        load_skills(settings.skills_dir, registry)
    app.state.app_state = AppState(
        store=store, llm=_build_llm(settings), default_model=settings.default_model,
        soul=soul, registry=registry,
    )
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_builder_tools.py -q`
Expected: PASS (4 tests).

- [ ] **Step 8: Run the full app suite + boot/isolation gates**

Run: `cd backend && python -m pytest ../tests/app -q`
Expected: PASS (everything; existing route/echo tests unaffected — they construct `AppState` without a registry, so `registry` defaults to empty and the agent gets no tools, exactly as before).

- [ ] **Step 9: Commit**

```bash
git add backend/app/builder.py backend/app/deps.py backend/app/routes/responses.py backend/app/lean_main.py tests/app/test_builder_tools.py
git commit -m "feat(app): wire ToolRegistry into build_context + boot (default tools + skills)"
```

---

## Self-Review (completed against the spec)

- **`ToolRegistry` (register/get/names/build_toolbox; unknown skipped)** → Task 1.
- **Default trio `current_datetime`/`web_fetch`/`web_search` + `build_default_registry` (search only when configured/injected); tools return error strings, never raise** → Task 2.
- **Skills loader (local `get_tools()`, broken skill skipped) + MCP adapter (schema→Tool, injected `call`, prefix namespacing)** → Task 3.
- **`build_context` selects via `soul.tools_enabled` (or all), builds the ToolBox, renders the prompt naming the actual tools; `registry=None` → empty (Plan-A behavior)** → Task 4.
- **`AppState.registry`, route + lean_main wiring (defaults + skills_dir), settings (search_*, skills_dir)** → Tasks 2 & 4.
- **MCP/search are interfaces (no live transport/vendor)** → honored; `_provider_from_settings` returns None with a log; MCP `call` is injected.
- **Import-lean + boot/isolation green; route-compatible (`registry` kwarg default)** → Global Constraints + Task 4.
- **End-to-end: a wired tool is dispatchable through `ctx.tools.dispatch`** → Task 4 `test_wired_tool_is_dispatchable` (the agent loop already advertises + dispatches, so this confirms the wiring without a tool-calling LLM fake).

No placeholders; signatures (`ToolRegistry.build_toolbox(names=None)`, `make_*_tool(...)`, `build_default_registry(settings, *, search_provider=None)`, `load_skills(path, registry)`, `mcp_tool_to_tool(spec, call, prefix="")`, `register_mcp_tools(...)`, `build_context(..., *, soul=DEFAULT_SOUL, registry=None)`) are consistent across tasks and with Plan A.
