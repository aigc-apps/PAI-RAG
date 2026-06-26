# Clean Tool / ToolBox (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `llama_index`-coupled `ToolBox` at the agent boundary with a clean `Tool`/`ToolBox` abstraction, plus an adapter so the existing RAG `FunctionTool`s keep working unchanged.

**Architecture:** A clean `Tool` (name, description, JSON-schema parameters, async fn, return_direct) and a `ToolBox` over it — no `llama_index` import in the agent core. A `Tool.from_function_tool(ft)` adapter bridges existing `llama_index` `FunctionTool`s so callers (agent_service) wrap their tool list at the boundary. Tracing + retry + dispatch semantics are preserved.

**Tech Stack:** Python 3.11, dataclasses, async, tenacity, pytest, OpenTelemetry (existing trace helpers).

**Reference spec:** `docs/superpowers/specs/2026-06-26-agent-api-protocol-design.md` (migration step 3). Phase 1 (AgentEvent + serializers) is done. This phase is internal cleanup; `Agent.run`/serializers are unchanged.

**Scope note:** The 12 RAG tool factories under `backend/tools/` are NOT changed — they still produce `FunctionTool`s; `agent_service` wraps them via the adapter. Decoupling those factories from `llama_index` is a later follow-up.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/agent/tools/__init__.py` (new) | re-export `Tool`, `ToolBox`, `ToolResult` (so `from agent.tools import ToolBox` keeps working) |
| `backend/agent/tools/base.py` (new, from `agent/tools.py`) | clean `Tool` + `ToolBox` + `ToolResult` + traced dispatch (no `llama_index`) |
| `backend/agent/tools.py` (delete) | replaced by the package |
| `backend/agent/tools/adapter.py` (new) | `Tool.from_function_tool(ft)` bridge for `llama_index` `FunctionTool` |
| `backend/service/agent/agent_service.py` (modify) | wrap the resolved `FunctionTool` list via the adapter before `ToolBox(...)` |
| `tests/agent/test_tools.py` (modify) | test the clean `Tool`/`ToolBox` with plain async fns (no `FunctionTool`) |
| `tests/agent/test_tool_adapter.py` (new) | test the `FunctionTool` → `Tool` adapter |
| `tests/agent/test_agent_run.py` (modify) | `_box` helper builds clean `Tool`s instead of `FunctionTool` |

---

## Task 1: clean `Tool` / `ToolBox` (no llama_index)

Convert the `agent/tools.py` module into a package and rewrite `ToolBox` to operate on a clean `Tool`. READ `backend/agent/tools.py` (current) and `backend/extensions/trace/pai_agent_wrapper.py` `instrument_async_call` (to mirror the per-tool span attributes in a clean tracer).

**Files:**
- Create: `backend/agent/tools/__init__.py`, `backend/agent/tools/base.py`
- Delete: `backend/agent/tools.py`
- Test: `tests/agent/test_tools.py` (rewrite)

- [ ] **Step 1: Rewrite tests `tests/agent/test_tools.py` to use clean Tools**

```text
import sys, os, json, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.tools import Tool, ToolBox
from agent.message import ToolCall


def _tool(fn, name, return_direct=False):
    return Tool(name=name, description=name, parameters={"type": "object", "properties": {}},
                fn=fn, return_direct=return_direct)


def test_dispatch_runs_tool_and_wraps_result():
    async def echo(x: str): return f"got {x}"
    box = ToolBox([_tool(echo, "echo")])
    res = asyncio.run(box.dispatch(ToolCall(id="c1", name="echo", arguments=json.dumps({"x": "hi"}))))
    assert res.ok and "got hi" in res.content
    assert res.message.role == "tool" and res.message.tool_call_id == "c1"


def test_dispatch_unknown_tool_is_error_not_crash():
    async def echo(x: str): return x
    box = ToolBox([_tool(echo, "echo")])
    res = asyncio.run(box.dispatch(ToolCall(id="c2", name="nope", arguments="{}")))
    assert not res.ok and "Unknown tool" in res.message.content


def test_dispatch_tool_exception_becomes_error_result():
    async def boom(): raise RuntimeError("kaboom")
    box = ToolBox([_tool(boom, "boom")])
    res = asyncio.run(box.dispatch(ToolCall(id="c3", name="boom", arguments="{}")))
    assert not res.ok and "kaboom" in res.error


def test_is_return_direct_flag():
    async def faq(q: str): return q
    box = ToolBox([_tool(faq, "faq", return_direct=True)])
    assert box.is_return_direct("faq") is True
    assert box.is_return_direct("missing") is False


def test_openai_schema_lists_tools():
    async def echo(x: str): return x
    box = ToolBox([_tool(echo, "echo")])
    schema = box.openai_schema()
    assert schema[0]["type"] == "function"
    assert schema[0]["function"]["name"] == "echo"
    assert schema[0]["function"]["parameters"]["type"] == "object"
```

- [ ] **Step 2: Run to verify failure** — `cd backend && python -m pytest ../tests/agent/test_tools.py -v` → `ImportError`/`ModuleNotFoundError`.

- [ ] **Step 3: Create the package.** `git mv backend/agent/tools.py backend/agent/tools/base.py` then create `backend/agent/tools/__init__.py`:

```text
from agent.tools.base import Tool, ToolBox, ToolResult

__all__ = ["Tool", "ToolBox", "ToolResult"]
```

Rewrite `backend/agent/tools/base.py` (replace its contents):

```text
from __future__ import annotations
import traceback
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional
from tenacity import RetryError, retry, stop_after_attempt, wait_fixed
from loguru import logger
from agent.message import Message, ToolCall
from utils.json_utils import parse_tool_arguments


@dataclass
class Tool:
    """A callable the agent can invoke. `fn` is an async callable that takes the
    parsed JSON arguments as kwargs and returns a string (the tool output)."""
    name: str
    description: str
    parameters: Dict          # JSON Schema for the arguments object
    fn: Callable[..., Awaitable[str]]
    return_direct: bool = False

    def openai_schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters,
        }}


@dataclass
class ToolResult:
    message: Message
    content: Optional[str]
    error: Optional[str]
    name: str
    tool_call: ToolCall

    @property
    def ok(self) -> bool:
        return self.error is None


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def _call_with_retry(tool: Tool, args: dict) -> str:
    """Invoke the tool fn inside a tracing span, with retry. Mirrors the span
    attributes the old instrument_async_call set (tool name + args)."""
    from extensions.trace.base import get_tracer
    with get_tracer().start_as_current_span(f"tool {tool.name}") as span:
        try:
            span.set_attribute("tool.name", tool.name)
        except Exception:
            pass
        result = await tool.fn(**args)
        return result if isinstance(result, str) else str(result)


class ToolBox:
    def __init__(self, tools: List[Tool]):
        self.tools = tools
        self._by_name = {t.name: t for t in tools}

    def __bool__(self) -> bool:
        return bool(self.tools)

    def get(self, name: str) -> Optional[Tool]:
        return self._by_name.get(name)

    def is_return_direct(self, name: str) -> bool:
        tool = self._by_name.get(name)
        return bool(tool and tool.return_direct)

    def openai_schema(self) -> List[dict]:
        return [t.openai_schema() for t in self.tools]

    async def dispatch(self, tc: ToolCall) -> ToolResult:
        tool = self._by_name.get(tc.name)
        if tool is None:
            err = f"Unknown tool: {tc.name}. Available: {list(self._by_name)}"
            logger.warning(err)
            return ToolResult(message=Message("tool", content=err, tool_call_id=tc.id),
                              content=None, error=err, name=tc.name, tool_call=tc)
        args = parse_tool_arguments(tc.arguments)
        logger.info(f"Calling tool {tc.name} with args: {args}")
        try:
            content = await _call_with_retry(tool, args)
            return ToolResult(message=Message("tool", content=content, tool_call_id=tc.id),
                              content=content, error=None, name=tc.name, tool_call=tc)
        except RetryError as re:
            logger.error(f"Tool call failed after retries: {traceback.format_exc()}")
            err = f"Tool call failed: {re.last_attempt.exception()}"
        except Exception as ex:
            logger.error(f"Tool call failed: {traceback.format_exc()}")
            err = f"Tool call failed: {ex}"
        return ToolResult(message=Message("tool", content=err, tool_call_id=tc.id),
                          content=None, error=err, name=tc.name, tool_call=tc)
```

If `get_tracer` is not at `extensions.trace.base`, grep for it (`grep -rn "def get_tracer" backend`) and import from the right place; if no clean tracer is readily available, fall back to calling `tool.fn(**args)` directly inside `_call_with_retry` (no span) and note it — tracing-per-tool is a nice-to-have, correctness is the bar.

- [ ] **Step 4: Run to verify pass** — `cd backend && python -m pytest ../tests/agent/test_tools.py -v` → 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/agent/tools/ tests/agent/test_tools.py
git rm backend/agent/tools.py 2>/dev/null || true
git commit -m "feat(agent): clean Tool/ToolBox abstraction (no llama_index)"
```

---

## Task 2: `FunctionTool` → `Tool` adapter

**Files:**
- Create: `backend/agent/tools/adapter.py`
- Test: `tests/agent/test_tool_adapter.py`

- [ ] **Step 1: Write failing tests**

```text
import sys, os, json, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from llama_index.core.tools import FunctionTool
from agent.tools import ToolBox
from agent.tools.adapter import tool_from_function_tool
from agent.message import ToolCall


def test_adapter_preserves_name_schema_and_dispatch():
    async def echo(x: str): return f"echoed {x}"
    ft = FunctionTool.from_defaults(async_fn=echo, name="echo")
    tool = tool_from_function_tool(ft)
    assert tool.name == "echo"
    schema = tool.openai_schema()
    assert schema["function"]["name"] == "echo" and "x" in json.dumps(schema["function"]["parameters"])
    box = ToolBox([tool])
    res = asyncio.run(box.dispatch(ToolCall(id="c1", name="echo", arguments=json.dumps({"x": "hi"}))))
    assert res.ok and "echoed hi" in res.content


def test_adapter_carries_return_direct():
    async def faq(q: str): return q
    ft = FunctionTool.from_defaults(async_fn=faq, name="faq", return_direct=True)
    assert tool_from_function_tool(ft).return_direct is True
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: agent.tools.adapter`.

- [ ] **Step 3: Implement `backend/agent/tools/adapter.py`**

READ how `FunctionTool.metadata` exposes the schema: `ft.metadata.name`, `ft.metadata.description`, `ft.metadata.to_openai_tool(skip_length_check=True)["function"]["parameters"]` (the JSON-schema parameters), `ft.metadata.return_direct`, and `ft.acall(**kwargs)` (returns a `ToolOutput` with `.content`).

```text
from __future__ import annotations
from llama_index.core.tools.function_tool import FunctionTool
from agent.tools.base import Tool


def tool_from_function_tool(ft: FunctionTool) -> Tool:
    """Adapt a llama_index FunctionTool into a clean Tool, preserving name,
    description, JSON-schema parameters, return_direct, and async dispatch."""
    meta = ft.metadata
    params = meta.to_openai_tool(skip_length_check=True)["function"].get("parameters", {"type": "object", "properties": {}})

    async def _fn(**kwargs) -> str:
        out = await ft.acall(**kwargs)
        return out.content if hasattr(out, "content") else str(out)

    return Tool(
        name=meta.name,
        description=meta.description or meta.name,
        parameters=params,
        fn=_fn,
        return_direct=getattr(meta, "return_direct", False),
    )
```

If `to_openai_tool`'s structure differs, adapt to extract the parameters JSON schema (grep `to_openai_tool` usage / read the metadata class) and report. Optionally re-export `tool_from_function_tool` from `agent/tools/__init__.py`.

- [ ] **Step 4: Run to verify pass** — `cd backend && python -m pytest ../tests/agent/test_tool_adapter.py -v` → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/agent/tools/adapter.py tests/agent/test_tool_adapter.py backend/agent/tools/__init__.py
git commit -m "feat(agent): FunctionTool -> clean Tool adapter"
```

---

## Task 3: wire the adapter at the agent_service boundary + fix loop tests

**Files:**
- Modify: `backend/service/agent/agent_service.py`
- Modify: `tests/agent/test_agent_run.py`

- [ ] **Step 1: Wrap FunctionTools at the boundary in `agent_service.py`**

Find where `create_agent` builds `ToolBox(tools)` (the `tools` is a `List[FunctionTool]` from the RAG factories). Change it to wrap each via the adapter:

```text
from agent.tools.adapter import tool_from_function_tool
...
tools=ToolBox([tool_from_function_tool(t) for t in tools]),
```

Grep for any OTHER construction of `ToolBox(...)` in non-test backend code (`grep -rn "ToolBox(" backend --include="*.py" | grep -v tests | grep -v graphify`) and wrap those FunctionTool lists too. Do NOT change the `tools/` factories themselves.

- [ ] **Step 2: Update the loop tests' `_box` helper to clean Tools**

In `tests/agent/test_agent_run.py`, the `_box` helper currently builds a `FunctionTool` and passes it to `ToolBox`. Replace it so it builds a clean `Tool` directly (drop the `llama_index` import there if now unused):

```text
from agent.tools import Tool, ToolBox

def _box(fn, name, return_direct=False):
    return ToolBox([Tool(name=name, description=name,
                         parameters={"type": "object", "properties": {}},
                         fn=fn, return_direct=return_direct)])
```

Note: some `_box` test fns take args (e.g. `echo(x)`), some take none (`faq()`, `echo()`); the clean `Tool.fn` is called with the parsed kwargs, so `dispatch(ToolCall(arguments="{}"))` calls `fn()` with no kwargs — matches the existing test fns. Keep the test bodies otherwise unchanged.

- [ ] **Step 3: Run the full agent + protocol suite**

Run: `cd backend && python -c "import service.agent.agent_service, api.v1.chat" && python -m pytest ../tests/agent/ ../tests/api/protocol/ -q`
Expected: imports clean; all pass.

- [ ] **Step 4: Confirm the agent core no longer imports llama_index**

Run: `grep -rn "llama_index" backend/agent --include="*.py"`
Expected: matches ONLY in `backend/agent/tools/adapter.py` (the bridge) — nowhere else in `agent/`. If `agent/agent.py` or others still import it, that's a miss; report it.

- [ ] **Step 5: Commit**

```bash
git add backend/service/agent/agent_service.py tests/agent/test_agent_run.py
git commit -m "refactor(agent): wrap RAG FunctionTools via clean-Tool adapter at the boundary"
```

---

## Notes for the implementer

- The clean `Tool.fn` returns a **string**; `ToolBox.dispatch` wraps it in `ToolResult.content`. The adapter's `_fn` extracts `.content` from the FunctionTool's `ToolOutput` so adapted tools still return strings.
- Per-tool tracing: the old `ToolBox` used `instrument_async_call(ft, args)` (FunctionTool-specific). The clean `_call_with_retry` uses a plain OTel span. If you can't cleanly get a tracer, omit the span (correctness over tracing) and note it — the run-level `@pai_agent_wrapper` span still exists.
- Do NOT touch `Agent.run`, the serializers, or the `tools/` RAG factories. This phase only swaps the tool abstraction + adds the boundary adapter.
- After Task 3, `from agent.tools import ToolBox` (used by `agent/agent.py` and `agent_service`) still resolves via the new package `__init__.py` — verify no importer breaks.
