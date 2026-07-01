# Agent Loop Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 240-line `react_agent.run_async` with a thin `Agent` built around `Message` / `AgentContext` / a single `build_messages` seam, so "what does the model receive?" is answered in one pure, logged function.

**Architecture:** New `backend/agent/` modules — `message.py` (one normalized message type + thread normalization), `context.py` (`AgentContext` assembled once by the caller), `tools.py` (`ToolBox` over existing `FunctionTool`s), `budgeting.py` (renamed `message_manager`), `agent.py` (`build_messages` + a flat call→dispatch→append loop). Intent-nudge buffering is dropped. Budgeting / `return_direct` / guardrails / streaming / tracing are preserved as separate steps the loop calls or the caller wraps.

**Tech Stack:** Python 3.11, async generators, pytest, llama_index `FunctionTool`, OpenAI chat-completion message dicts, loguru.

**Reference spec:** `docs/superpowers/specs/2026-06-25-agent-loop-refactor-design.md`

**Key implementation decision (refines spec):** The loop holds `list[Message]`. Budgeting and the LLM client both speak OpenAI **wire dicts**, so conversion is isolated at exactly two boundaries: `budgeting.fit(list[Message]) -> list[Message]` wraps the existing dict-based manager internally, and `_stream_turn` serializes via `m.to_wire()` right before `llm.astream`. This keeps `AgentMessageManager`'s internals and `llm.astream`'s dict contract untouched.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/agent/message.py` (new) | `ToolCall`, `Message`, `from_thread()`, `keep_last_rounds()` |
| `backend/agent/context.py` (rewrite) | `RunVars`, `Attachment`, `AgentContext` |
| `backend/agent/budgeting.py` (renamed from `message_manager.py`) | existing manager + `fit(list[Message])` wrapper |
| `backend/agent/tools.py` (new) | `ToolBox`, `ToolResult` |
| `backend/agent/agent.py` (new) | `Agent.build_messages()`, `Agent.run()`, `_stream_turn()` |
| `backend/agent/react_agent.py` (delete, Task 9) | — |
| `backend/agent/state.py` (delete, Task 9) | — |
| `backend/service/agent/agent_service.py` (modify) | build `AgentContext`, return attachments as data |
| `backend/api/v1/chat.py` (modify) | assemble `AgentContext`, guardrails at edges |
| `backend/tests/agent/*` (new) | unit tests incl. `FakeLLM` |

Tests live in `tests/agent/` mirroring existing `tests/service/agent/` conventions (`sys.path.insert` to `backend`).

---

## Task 1: `Message` type + wire conversion

**Files:**
- Create: `backend/agent/message.py`
- Test: `tests/agent/test_message.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agent/test_message.py
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from agent.message import Message, ToolCall


def test_text_message_roundtrips_to_wire():
    m = Message(role="user", content="hi")
    assert m.to_wire() == {"role": "user", "content": "hi"}
    assert Message.from_wire({"role": "user", "content": "hi"}) == m


def test_assistant_tool_call_to_wire():
    m = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="c1", name="read", arguments='{"x":1}')],
    )
    wire = m.to_wire()
    assert wire["role"] == "assistant"
    assert wire["content"] is None
    assert wire["tool_calls"][0] == {
        "id": "c1",
        "type": "function",
        "function": {"name": "read", "arguments": '{"x":1}'},
    }


def test_tool_result_message_to_wire():
    m = Message(role="tool", content="result text", tool_call_id="c1")
    assert m.to_wire() == {
        "role": "tool",
        "content": "result text",
        "tool_call_id": "c1",
    }


def test_list_content_preserved_for_multimodal():
    parts = [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "http://x"}},
    ]
    m = Message(role="user", content=parts)
    assert m.to_wire()["content"] == parts
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest ../tests/agent/test_message.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.message'`

- [ ] **Step 3: Implement `message.py` (type + wire conversion)**

```python
# backend/agent/message.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union

ContentPart = dict  # {"type": "text"|"image_url", ...} for multimodal turns


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, as the model emitted it


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: Union[str, List[ContentPart], None] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_wire(self) -> dict:
        msg: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    @classmethod
    def from_wire(cls, d: dict) -> "Message":
        raw_tcs = d.get("tool_calls") or []
        tool_calls = [
            ToolCall(
                id=tc.get("id", ""),
                name=tc.get("function", {}).get("name", ""),
                arguments=tc.get("function", {}).get("arguments", "") or "",
            )
            for tc in raw_tcs
        ] or None
        return cls(
            role=d.get("role", ""),
            content=d.get("content"),
            tool_calls=tool_calls,
            tool_call_id=d.get("tool_call_id"),
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && python -m pytest ../tests/agent/test_message.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/agent/message.py tests/agent/test_message.py
git commit -m "feat(agent): add normalized Message type with wire conversion"
```

---

## Task 2: `from_thread` normalization + `keep_last_rounds`

Port the logic from `agent/state.py::convert_thread_messages` and `_keep_last_n_rounds` into typed form. Read `backend/agent/state.py` lines 34–121 and 158–170 for the exact rules being ported (assistant tool-call parts → assistant+tool message pairs, user content arrays → flattened text or text+image, orphaned tool messages dropped).

**Files:**
- Modify: `backend/agent/message.py`
- Test: `tests/agent/test_message.py`

- [ ] **Step 1: Add failing tests**

```python
# append to tests/agent/test_message.py
from agent.message import from_thread, keep_last_rounds


def test_from_thread_plain_user_and_assistant():
    out = from_thread(
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
    )
    assert [m.role for m in out] == ["user", "assistant"]
    assert out[0].content == "q1"


def test_from_thread_flattens_user_content_array_text_only():
    out = from_thread(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "line1"},
                    {"type": "text", "text": "line2"},
                ],
            }
        ]
    )
    assert out[0].content == "line1\nline2"


def test_from_thread_keeps_image_parts():
    out = from_thread(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "http://x"}},
                ],
            }
        ]
    )
    assert isinstance(out[0].content, list)
    assert out[0].content[0] == {"type": "text", "text": "look"}


def test_from_thread_drops_orphan_tool_message():
    # tool message with no preceding assistant tool_call is dropped
    out = from_thread(
        [{"role": "tool", "content": "x", "tool_call_id": "missing"}]
    )
    assert out == []


def test_keep_last_rounds_trims_by_user_turn():
    msgs = [
        Message("user", "u1"),
        Message("assistant", "a1"),
        Message("user", "u2"),
        Message("assistant", "a2"),
    ]
    assert keep_last_rounds(msgs, 1) == msgs[2:]


def test_keep_last_rounds_zero_is_noop():
    msgs = [Message("user", "u1")]
    assert keep_last_rounds(msgs, 0) == msgs
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest ../tests/agent/test_message.py -v`
Expected: FAIL — `ImportError: cannot import name 'from_thread'`

- [ ] **Step 3: Implement `from_thread` + `keep_last_rounds`**

```python
# append to backend/agent/message.py


def _has_preceding_tool_call(result: List[Message], tool_call_id: str) -> bool:
    for msg in reversed(result):
        if msg.role == "tool":
            continue
        if msg.role == "assistant" and msg.tool_calls:
            return any(tc.id == tool_call_id for tc in msg.tool_calls)
        return False
    return False


def from_thread(raw: List[dict]) -> List[Message]:
    """Normalize an incoming thread (mixed dict shapes) into Messages.

    - user content arrays → flattened text, or text+image parts kept for vision
    - assistant 'tool-call' content parts → assistant(tool_calls) + tool result pairs
    - tool messages without a matching preceding assistant tool_call → dropped
    """
    result: List[Message] = []
    for d in raw:
        role = d.get("role", "")
        content = d.get("content")

        if role == "tool":
            tcid = d.get("tool_call_id", "")
            if tcid and _has_preceding_tool_call(result, tcid):
                result.append(Message.from_wire(d))
            continue

        if role == "assistant" and d.get("tool_calls"):
            result.append(Message.from_wire(d))
            continue

        if not isinstance(content, list):
            result.append(Message.from_wire(d))
            continue

        if role == "user":
            text_parts, other_parts = [], []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") in (
                    "image_url",
                    "image",
                ):
                    other_parts.append(part)
            if other_parts:
                new_content = list(other_parts)
                if text_parts:
                    new_content.insert(
                        0, {"type": "text", "text": "\n".join(text_parts)}
                    )
                result.append(Message(role="user", content=new_content))
            else:
                result.append(
                    Message(role="user", content="\n".join(text_parts))
                )
            continue

        if role == "assistant":
            tc_parts, text_parts = [], []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "tool-call":
                    tc_parts.append(part)
                elif (
                    part.get("type") == "text"
                    and (part.get("text") or "").strip()
                ):
                    text_parts.append(part["text"])
            for tc in tc_parts:
                import json

                args = tc.get("args", {})
                args_str = (
                    json.dumps(args, ensure_ascii=False)
                    if isinstance(args, dict)
                    else str(args or "{}")
                )
                result.append(
                    Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id=tc.get("toolCallId", ""),
                                name=tc.get("toolName", ""),
                                arguments=args_str,
                            )
                        ],
                    )
                )
                tool_result = tc.get("result", "")
                import json as _json

                result.append(
                    Message(
                        role="tool",
                        tool_call_id=tc.get("toolCallId", ""),
                        content=tool_result
                        if isinstance(tool_result, str)
                        else _json.dumps(tool_result, ensure_ascii=False),
                    )
                )
            if text_parts:
                result.append(
                    Message(role="assistant", content="\n".join(text_parts))
                )
            continue

        result.append(Message.from_wire(d))
    return result


def keep_last_rounds(msgs: List[Message], n: int) -> List[Message]:
    if n <= 0:
        return msgs
    user_idx = [i for i, m in enumerate(msgs) if m.role == "user"]
    if len(user_idx) <= n:
        return msgs
    return msgs[user_idx[-n] :]
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && python -m pytest ../tests/agent/test_message.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/agent/message.py tests/agent/test_message.py
git commit -m "feat(agent): typed thread normalization (from_thread, keep_last_rounds)"
```

---

## Task 3: `context.py` — `RunVars`, `Attachment`, `AgentContext`

**Files:**
- Modify (rewrite): `backend/agent/context.py`
- Test: `tests/agent/test_context.py`

- [ ] **Step 1: Write failing test**

```python
# tests/agent/test_context.py
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.context import RunVars, Attachment, AgentContext
from agent.message import Message


def test_runvars_autofills_datetime():
    assert RunVars().current_datetime  # non-empty


def test_agent_context_holds_assembly_inputs():
    ctx = AgentContext(
        system_prompt="sys",
        history=[Message("user", "old")],
        current_turn=Message("user", "now"),
        attachments=[Attachment(name="r.pdf", body="text")],
        hints=["use search-file-chunks"],
        tools=None,
        run_vars=RunVars(current_datetime="2026-06-25"),
    )
    assert ctx.current_turn.content == "now"
    assert ctx.attachments[0].name == "r.pdf"
    assert ctx.hints == ["use search-file-chunks"]
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest ../tests/agent/test_context.py -v`
Expected: FAIL — `ImportError: cannot import name 'AgentContext'`

- [ ] **Step 3: Implement `context.py`**

```python
# backend/agent/context.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from utils.time_utils import get_current_time_str
from agent.message import Message


@dataclass
class RunVars:
    """Runtime values rendered into the current turn (e.g. the time header)."""

    current_datetime: str = field(default_factory=get_current_time_str)


@dataclass
class Attachment:
    """Resolved attachment content to inject inline (file text or a status note)."""

    name: str
    body: str


@dataclass
class AgentContext:
    """Everything needed to build model input and run, assembled ONCE by the caller.
    This is the single inspectable object: log it and you know what the model sees.
    """

    system_prompt: str
    history: List[Message]
    current_turn: Message
    attachments: List[Attachment]
    hints: List[
        str
    ]  # instructional text (search-chunks hint, no-question summarize hint)
    tools: "object"  # ToolBox; typed loosely to avoid an import cycle
    run_vars: RunVars
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && python -m pytest ../tests/agent/test_context.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/agent/context.py tests/agent/test_context.py
git commit -m "feat(agent): AgentContext/RunVars/Attachment — single assembly input"
```

---

## Task 4: `budgeting.py` — rename + `Message` wrapper

The 375-line `AgentMessageManager` is preserved verbatim; we only rename the module and add a thin `fit(list[Message]) -> list[Message]` wrapper so the loop stays in `Message` while the dict-based internals are untouched.

**Files:**
- Rename: `backend/agent/message_manager.py` → `backend/agent/budgeting.py`
- Modify: `backend/agent/budgeting.py` (add wrapper)
- Test: `tests/agent/test_budgeting.py`

- [ ] **Step 1: Rename the module (no logic change)**

```bash
git mv backend/agent/message_manager.py backend/agent/budgeting.py
```

- [ ] **Step 2: Write failing test for the wrapper**

```python
# tests/agent/test_budgeting.py
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.budgeting import AgentMessageManager
from agent.message import Message


def test_fit_returns_messages_and_keeps_short_history():
    mgr = AgentMessageManager(context_window=110000, max_output_tokens=8000)
    msgs = [Message("system", "s"), Message("user", "hello")]
    out = mgr.fit(msgs)
    assert all(isinstance(m, Message) for m in out)
    assert out[-1].content == "hello"
```

- [ ] **Step 3: Run to verify failure**

Run: `cd backend && python -m pytest ../tests/agent/test_budgeting.py -v`
Expected: FAIL — `AttributeError: 'AgentMessageManager' object has no attribute 'fit'`

- [ ] **Step 4: Add the wrapper method to `AgentMessageManager`**

```python
# add inside class AgentMessageManager in backend/agent/budgeting.py
def fit(self, messages):
    """Message-typed wrapper around the dict-based fit_to_budget."""
    from agent.message import Message

    wire = [m.to_wire() for m in messages]
    fitted = self.fit_to_budget(wire)
    return [Message.from_wire(d) for d in fitted]
```

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && python -m pytest ../tests/agent/test_budgeting.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Fix the one existing importer and run repo tests**

`react_agent.py` still imports `from agent.message_manager import AgentMessageManager`. It will be deleted in Task 9, but keep the repo importable now: update that import to `from agent.budgeting import AgentMessageManager`.

Run: `cd backend && python -c "import agent.react_agent"`
Expected: no error.

- [ ] **Step 7: Commit**

```bash
git add backend/agent/budgeting.py backend/agent/react_agent.py tests/agent/test_budgeting.py
git commit -m "refactor(agent): rename message_manager->budgeting, add Message fit() wrapper"
```

---

## Task 5: `tools.py` — `ToolBox` + `ToolResult`

Consolidates `tool_fn_map`, `execute_single_tool_call`, `call_tool_with_retry`, `to_openai_tool`, and `check_and_handle_return_direct` (from `react_agent.py` and `tool_utils.py`).

**Files:**
- Create: `backend/agent/tools.py`
- Test: `tests/agent/test_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agent/test_tools.py
import sys, os, json, asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from llama_index.core.tools import FunctionTool
from agent.tools import ToolBox
from agent.message import ToolCall


def _box(fn, name, return_direct=False):
    tool = FunctionTool.from_defaults(
        async_fn=fn, name=name, return_direct=return_direct
    )
    return ToolBox([tool])


def test_dispatch_runs_tool_and_wraps_result():
    async def echo(x: str):
        return f"got {x}"

    box = _box(echo, "echo")
    tc = ToolCall(id="c1", name="echo", arguments=json.dumps({"x": "hi"}))
    res = asyncio.run(box.dispatch(tc))
    assert res.ok and "got hi" in res.content
    assert res.message.role == "tool" and res.message.tool_call_id == "c1"


def test_dispatch_unknown_tool_is_error_not_crash():
    async def echo(x: str):
        return x

    box = _box(echo, "echo")
    res = asyncio.run(
        box.dispatch(ToolCall(id="c2", name="nope", arguments="{}"))
    )
    assert not res.ok and "Unknown tool" in res.message.content


def test_is_return_direct_flag():
    async def faq(q: str):
        return q

    box = _box(faq, "faq", return_direct=True)
    assert box.is_return_direct("faq") is True
    assert box.is_return_direct("missing") is False


def test_openai_schema_lists_tools():
    async def echo(x: str):
        return x

    box = _box(echo, "echo")
    schema = box.openai_schema()
    assert schema[0]["function"]["name"] == "echo"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest ../tests/agent/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.tools'`

- [ ] **Step 3: Implement `tools.py`**

```python
# backend/agent/tools.py
from __future__ import annotations
import traceback
from dataclasses import dataclass
from typing import List, Optional
from llama_index.core.tools.function_tool import FunctionTool
from tenacity import RetryError, retry, stop_after_attempt, wait_fixed
from loguru import logger
from agent.message import Message, ToolCall
from utils.json_utils import parse_tool_arguments


@dataclass
class ToolResult:
    message: Message  # the "tool" role message to append to history
    content: Optional[str]  # raw tool output (None on error)
    error: Optional[str]  # error text (None on success)
    name: str
    tool_call: ToolCall

    @property
    def ok(self) -> bool:
        return self.error is None


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def _call_with_retry(async_fn, fn_args):
    from extensions.trace.pai_agent_wrapper import instrument_async_call

    return await instrument_async_call(async_fn, fn_args)


class ToolBox:
    def __init__(self, tools: List[FunctionTool]):
        self.tools = tools
        self._by_name = {t.metadata.name: t for t in tools}

    def __bool__(self) -> bool:
        return bool(self.tools)

    def get(self, name: str) -> Optional[FunctionTool]:
        return self._by_name.get(name)

    def is_return_direct(self, name: str) -> bool:
        tool = self._by_name.get(name)
        return bool(tool and getattr(tool.metadata, "return_direct", False))

    def openai_schema(self) -> List[dict]:
        return [
            t.metadata.to_openai_tool(skip_length_check=True)
            for t in self.tools
        ]

    async def dispatch(self, tc: ToolCall) -> ToolResult:
        tool = self._by_name.get(tc.name)
        if tool is None:
            err = f"Unknown tool: {tc.name}. Available: {list(self._by_name)}"
            logger.warning(err)
            return ToolResult(
                message=Message("tool", content=err, tool_call_id=tc.id),
                content=None,
                error=err,
                name=tc.name,
                tool_call=tc,
            )
        args = parse_tool_arguments(tc.arguments)
        logger.info(f"Calling tool {tc.name} with args: {args}")
        try:
            out = await _call_with_retry(tool.async_fn, args)
            content = out.content
            return ToolResult(
                message=Message("tool", content=content, tool_call_id=tc.id),
                content=content,
                error=None,
                name=tc.name,
                tool_call=tc,
            )
        except RetryError as re:
            err = f"Tool call failed: {re.last_attempt.exception()}"
        except Exception as ex:
            logger.error(f"Tool call failed: {traceback.format_exc()}")
            err = f"Tool call failed: {ex}"
        return ToolResult(
            message=Message("tool", content=err, tool_call_id=tc.id),
            content=None,
            error=err,
            name=tc.name,
            tool_call=tc,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && python -m pytest ../tests/agent/test_tools.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/agent/tools.py tests/agent/test_tools.py
git commit -m "feat(agent): ToolBox (schema, dispatch+retry, return_direct)"
```

---

## Task 6: `build_messages` + `render_current_turn`

The single assembly point. Replaces the time-prefix logic in `react_agent.run_async` (lines 150–164) and the message mutation in `parse_attachment_tools`.

**Files:**
- Create: `backend/agent/agent.py` (start the file here)
- Test: `tests/agent/test_build_messages.py`

- [ ] **Step 1: Write failing tests (the bug-locking tests)**

```python
# tests/agent/test_build_messages.py
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.agent import Agent
from agent.context import AgentContext, RunVars, Attachment
from agent.message import Message


def _ctx(**kw):
    base = dict(
        system_prompt="SYS",
        history=[],
        current_turn=Message("user", "学生经验怎么样"),
        attachments=[],
        hints=[],
        tools=None,
        run_vars=RunVars(current_datetime="2026-06-25 18:00"),
    )
    base.update(kw)
    return AgentContext(**base)


def test_system_prompt_is_first_message():
    msgs = Agent.build_messages(_ctx())
    assert msgs[0].role == "system" and msgs[0].content == "SYS"


def test_current_turn_gets_time_prefix():
    msgs = Agent.build_messages(_ctx())
    assert "[System Time: 2026-06-25 18:00]" in msgs[-1].content
    assert "学生经验怎么样" in msgs[-1].content


def test_attachment_text_lands_in_model_input():
    # THE regression lock: attached file content must reach the model.
    msgs = Agent.build_messages(
        _ctx(attachments=[Attachment(name="resume.pdf", body="3年经验")])
    )
    text = msgs[-1].content
    assert '<attached_file name="resume.pdf">' in text
    assert "3年经验" in text


def test_hints_appended_after_attachments():
    msgs = Agent.build_messages(
        _ctx(
            attachments=[Attachment(name="r.pdf", body="x")],
            hints=["对于较长的文件可调用 search-file-chunks"],
        )
    )
    assert "search-file-chunks" in msgs[-1].content


def test_history_precedes_current_turn():
    msgs = Agent.build_messages(
        _ctx(history=[Message("user", "old"), Message("assistant", "reply")])
    )
    roles = [m.role for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]


def test_multimodal_current_turn_prefixes_text_part():
    turn = Message(
        "user",
        [
            {"type": "text", "text": "看图"},
            {"type": "image_url", "image_url": {"url": "http://x"}},
        ],
    )
    msgs = Agent.build_messages(_ctx(current_turn=turn))
    parts = msgs[-1].content
    assert isinstance(parts, list)
    assert "[System Time:" in parts[0]["text"] and "看图" in parts[0]["text"]
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest ../tests/agent/test_build_messages.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.agent'`

- [ ] **Step 3: Implement `agent.py` with `build_messages` (loop added in Task 7)**

```python
# backend/agent/agent.py
from __future__ import annotations
from typing import List
from loguru import logger
from agent.context import AgentContext, Attachment, RunVars
from agent.message import Message


def _format_attachments(attachments: List[Attachment]) -> str:
    if not attachments:
        return ""
    blocks = [
        f'<attached_file name="{a.name}">\n{a.body}\n</attached_file>'
        for a in attachments
    ]
    return "\n\n以下是用户本次上传的文件内容，请直接基于这些内容回答：\n\n" + "\n\n".join(blocks)


def render_current_turn(
    turn: Message,
    attachments: List[Attachment],
    hints: List[str],
    run_vars: RunVars,
) -> Message:
    """Assemble the live user turn: time header + user text + attachment blocks + hints.
    The ONLY place these are combined. Handles both str and multimodal-list content.
    """
    prefix = f"[System Time: {run_vars.current_datetime}]\n"
    suffix = _format_attachments(attachments)
    if hints:
        suffix += "\n\n" + "\n\n".join(hints)

    if isinstance(turn.content, list):
        parts = [dict(p) for p in turn.content]
        for p in parts:
            if p.get("type") == "text":
                p["text"] = prefix + (p.get("text") or "") + suffix
                break
        else:
            parts.insert(0, {"type": "text", "text": prefix + suffix})
        return Message(role="user", content=parts)

    base = turn.content or ""
    return Message(role="user", content=prefix + base + suffix)


class Agent:
    @staticmethod
    def build_messages(ctx: AgentContext) -> List[Message]:
        msgs: List[Message] = [Message("system", ctx.system_prompt)]
        msgs += ctx.history
        msgs.append(
            render_current_turn(
                ctx.current_turn, ctx.attachments, ctx.hints, ctx.run_vars
            )
        )
        logger.info(
            "[agent] model input: %d msgs; current turn head=%r",
            len(msgs),
            (
                msgs[-1].content
                if isinstance(msgs[-1].content, str)
                else "<multimodal>"
            )[:200],
        )
        return msgs
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && python -m pytest ../tests/agent/test_build_messages.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/agent/agent.py tests/agent/test_build_messages.py
git commit -m "feat(agent): build_messages — single model-input assembly point"
```

---

## Task 7: `Agent.run` loop + `_stream_turn`

The flat loop. Port the streaming/tool-call assembly from `react_agent.run_async` (lines 197–235 for chunk handling, 289–334 for dispatch) but WITHOUT the intent-nudge buffering (lines 178–269) and WITHOUT the in-loop time-prefix (now in `build_messages`).

**Files:**
- Modify: `backend/agent/agent.py`
- Create: `tests/agent/fake_llm.py`, `tests/agent/test_agent_run.py`

- [ ] **Step 1: Create the `FakeLLM` test double**

```python
# tests/agent/fake_llm.py
from common.llm.models import TextChunk
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)


def tool_call(idx, cid, name, args):
    return ChoiceDeltaToolCall(
        index=idx,
        id=cid,
        type="function",
        function=ChoiceDeltaToolCallFunction(name=name, arguments=args),
    )


class FakeLLM:
    """Yields scripted chunk lists, one per turn. context_window/max_tokens
    satisfy AgentMessageManager. Each 'turn' is a list[TextChunk]."""

    context_window = 110000
    max_tokens = 8000

    def __init__(self, turns):
        self._turns = list(turns)

    async def astream(self, messages, tools):
        chunks = self._turns.pop(0)

        async def gen():
            for c in chunks:
                yield c

        return gen()
```

- [ ] **Step 2: Write failing tests for the loop**

```python
# tests/agent/test_agent_run.py
import sys, os, asyncio, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from llama_index.core.tools import FunctionTool
from common.llm.models import TextChunk, ToolResultChunk
from agent.agent import Agent
from agent.tools import ToolBox
from agent.context import AgentContext, RunVars
from agent.message import Message
from fake_llm import FakeLLM, tool_call


def _ctx(tools, turn="hi"):
    return AgentContext(
        system_prompt="SYS",
        history=[],
        current_turn=Message("user", turn),
        attachments=[],
        hints=[],
        tools=tools,
        run_vars=RunVars(current_datetime="t"),
    )


def _collect(agent, ctx):
    async def run():
        return [c async for c in agent.run(ctx)]

    return asyncio.run(run())


def _box(fn, name, return_direct=False):
    return ToolBox(
        [
            FunctionTool.from_defaults(
                async_fn=fn, name=name, return_direct=return_direct
            )
        ]
    )


def test_plain_text_answer_streams_and_stops():
    llm = FakeLLM([[TextChunk(delta="hello "), TextChunk(delta="world")]])
    agent = Agent(llm, max_steps=5)

    async def echo(x: str):
        return x

    out = _collect(agent, _ctx(_box(echo, "echo")))
    text = "".join(c.delta for c in out if type(c) is TextChunk)
    assert text == "hello world"


def test_tool_call_then_final_answer():
    llm = FakeLLM(
        [
            [
                TextChunk(
                    tool_calls=[
                        tool_call(0, "c1", "echo", json.dumps({"x": "hi"}))
                    ]
                )
            ],
            [TextChunk(delta="done")],
        ]
    )
    agent = Agent(llm, max_steps=5)

    async def echo(x: str):
        return f"echoed {x}"

    out = _collect(agent, _ctx(_box(echo, "echo")))
    assert any(
        isinstance(c, ToolResultChunk) and "echoed hi" in (c.result or "")
        for c in out
    )
    assert "".join(c.delta for c in out if type(c) is TextChunk) == "done"


def test_return_direct_short_circuits():
    llm = FakeLLM([[TextChunk(tool_calls=[tool_call(0, "c1", "faq", "{}")])]])
    agent = Agent(llm, max_steps=5)

    async def faq():
        return json.dumps({"result": [{"content": "FAQ answer"}]})

    out = _collect(agent, _ctx(_box(faq, "faq", return_direct=True)))
    assert any("FAQ answer" in c.delta for c in out if type(c) is TextChunk)


def test_max_steps_emits_notice():
    # every turn asks for a tool again → never terminates on its own
    turns = [
        [TextChunk(tool_calls=[tool_call(0, f"c{i}", "echo", "{}")])]
        for i in range(3)
    ]
    llm = FakeLLM(turns)
    agent = Agent(llm, max_steps=2)

    async def echo():
        return "x"

    out = _collect(agent, _ctx(_box(echo, "echo")))
    assert any(
        "maximum" in c.delta.lower() or "max" in c.delta.lower()
        for c in out
        if type(c) is TextChunk
    )
```

- [ ] **Step 3: Run to verify failure**

Run: `cd backend && python -m pytest ../tests/agent/test_agent_run.py -v`
Expected: FAIL — `TypeError: Agent() takes no arguments` (no `__init__`/`run` yet)

- [ ] **Step 4: Implement `__init__`, `_stream_turn`, `run`**

```python
# add to backend/agent/agent.py
import asyncio
from typing import AsyncIterator, List, Optional, Tuple
from common.llm.models import (
    TextChunk,
    ReasoningChunk,
    ErrorChunk,
    ToolResultChunk,
)
from agent.tools import ToolBox
from agent.budgeting import AgentMessageManager
from utils.constants import try_get_int_env

MAX_RECURSION_STEPS = try_get_int_env("MAX_RECURSION_STEPS", 20)
LLM_STREAM_IDLE_TIMEOUT = try_get_int_env(
    "LLM_STREAM_IDLE_TIMEOUT_SECONDS", 30
)


async def _iter_with_idle_timeout(stream, timeout: int):
    it = stream.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(it.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return
        yield chunk


class Agent:
    def __init__(
        self,
        llm,
        max_steps: int = MAX_RECURSION_STEPS,
        budget: Optional[AgentMessageManager] = None,
    ):
        self.llm = llm
        self.max_steps = max_steps
        self.budget = budget or AgentMessageManager(
            context_window=getattr(llm, "context_window", 0),
            max_output_tokens=getattr(llm, "max_tokens", 0),
        )

    # build_messages / render_current_turn from Task 6 stay as-is.

    async def _stream_turn(self, messages, tools: ToolBox):
        """Stream one model turn. Yields TextChunk/ReasoningChunk live (no buffering).
        Returns (assistant_message, tool_calls)."""
        wire = [m.to_wire() for m in messages]
        stream = await self.llm.astream(
            messages=wire, tools=tools.openai_schema() if tools else []
        )
        text, tool_calls = "", []
        async for chunk in _iter_with_idle_timeout(
            stream, LLM_STREAM_IDLE_TIMEOUT
        ):
            if isinstance(chunk, ErrorChunk):
                self._error = chunk
                return None, []
            if chunk.tool_calls:
                tool_calls = chunk.tool_calls
            if isinstance(chunk, ReasoningChunk):
                yield chunk
            elif chunk.delta:
                text += chunk.delta
                yield TextChunk(delta=chunk.delta, usage=chunk.usage)
        self._last = (text, tool_calls)

    async def run(self, ctx: AgentContext) -> AsyncIterator[TextChunk]:
        from agent.message import ToolCall

        messages = self.build_messages(ctx)
        self._error = None
        for step in range(self.max_steps):
            messages = self.budget.fit(messages)
            self._last = ("", [])
            async for ev in self._stream_turn(messages, ctx.tools):
                yield ev
            if self._error is not None:
                yield self._error
                return
            text, raw_tcs = self._last

            if not raw_tcs:
                if text:
                    messages.append(Message("assistant", text))
                return  # plain text → done

            # Pair each raw ChoiceDeltaToolCall with its typed ToolCall so the
            # ToolResultChunk's `tool=` stays aligned even when some calls are
            # filtered as invalid (index into raw_tcs would otherwise misalign).
            pairs = [
                (
                    raw,
                    ToolCall(
                        id=raw.id,
                        name=raw.function.name,
                        arguments=raw.function.arguments or "",
                    ),
                )
                for raw in raw_tcs
                if raw.type == "function" and ctx.tools.get(raw.function.name)
            ]
            if not pairs:
                # no valid tool calls: record + let the model self-correct next step
                bad = raw_tcs[0]
                messages.append(
                    Message(
                        "assistant",
                        text or None,
                        tool_calls=[
                            ToolCall(
                                bad.id,
                                bad.function.name,
                                bad.function.arguments or "",
                            )
                        ],
                    )
                )
                messages.append(
                    Message(
                        "tool",
                        content=f"Error: tool '{bad.function.name}' not available.",
                        tool_call_id=bad.id,
                    )
                )
                continue

            for raw, _tc in pairs:
                yield TextChunk(tool_calls=[raw])
            for idx, (raw, tc) in enumerate(pairs):
                result = await ctx.tools.dispatch(tc)
                messages.append(
                    Message(
                        "assistant",
                        text if idx == 0 else None,
                        tool_calls=[tc],
                    )
                )
                capped = (
                    self.budget.cap_tool_result(result.message.content)
                    if result.message.content
                    else result.message.content
                )
                messages.append(
                    Message("tool", content=capped, tool_call_id=tc.id)
                )
                yield ToolResultChunk(
                    tool=raw, result=result.content, error=result.error
                )
                if ctx.tools.is_return_direct(tc.name) and result.ok:
                    direct = _format_return_direct(result.content)
                    if direct:
                        yield TextChunk(delta=direct)
                    return
        yield TextChunk(
            delta=f"\n\nReached maximum iteration count ({self.max_steps}), task ended."
        )


def _format_return_direct(content: Optional[str]) -> str:
    """Port of check_and_handle_return_direct's success formatting."""
    import json

    if not content:
        return "Tool call successful, but no content returned."
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "result" in data:
            parts = [
                it.get("content", "")
                for it in data.get("result", [])
                if isinstance(it, dict)
            ]
            joined = "\n\n".join(p for p in parts if p)
            return joined.strip() or content
    except Exception:
        pass
    return content
```

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && python -m pytest ../tests/agent/test_agent_run.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Run the whole new suite**

Run: `cd backend && python -m pytest ../tests/agent/ -v`
Expected: PASS (all)

- [ ] **Step 7: Commit**

```bash
git add backend/agent/agent.py tests/agent/fake_llm.py tests/agent/test_agent_run.py
git commit -m "feat(agent): flat run() loop + _stream_turn (no intent-nudge buffering)"
```

---

## Task 8: Migrate `agent_service` + `chat.py` to the new Agent

**Files:**
- Modify: `backend/service/agent/agent_service.py`
- Modify: `backend/api/v1/chat.py`
- Test: `tests/service/agent/test_agent_service.py` (existing helpers stay; update attachment path)

- [ ] **Step 1: `parse_attachment_tools` returns attachments + hints as DATA**

In `backend/service/agent/agent_service.py`, change `parse_attachment_tools` to stop calling `append_text(user_message, …)`. Instead accumulate and return:

```python
# return type becomes: (tools, cleanup, attachments: list[Attachment], hints: list[str])
from agent.context import Attachment

...
attachments: list[Attachment] = []
hints: list[str] = []
# where the code currently does append_text(user_message, _format_inline_attachments(inline_blocks)):
for name, body in inline_blocks:
    attachments.append(Attachment(name=name, body=body))
# no-question summarize instruction and search-file-chunks hint become:
if inline_blocks and not had_text:
    hints.append(
        "用户只上传了文件、没有文字提问。请阅读上述文件内容，推断用户意图，给出摘要或基于内容的有用回答；若需要额外信息再调用其他工具。"
    )
# (large_files branch) instead of append_text(... search hint ...):
hints.append(
    f"对于较长的文件 [{catalog_names}] 可以调用 `search-file-chunks` 工具按关键字检索更多内容。"
)
...
return attachment_tools, cleanup_code_sandbox, attachments, hints
```

Update `aget_tools` to thread these through and return them; `_format_inline_attachments` moves to `agent/agent.py` (already there as `_format_attachments`) — delete the agent_service copy.

- [ ] **Step 2: `create_agent` yields an `Agent` + the resolved context pieces**

Change `create_agent` to build and yield a small struct (or tuple) of `(Agent, system_prompt, tools: ToolBox, attachments, hints)` instead of a `ReactAgent`. Concretely:

```python
from agent.agent import Agent
from agent.tools import ToolBox

...
tools, sandbox_cleanup, attachments, hints = await self.aget_tools(...)
system_prompt = system_prompt.format(
    tools_str=_build_tools_summary(tools), context_str=""
)
agent = Agent(llm)
yield AgentBundle(
    agent=agent,
    system_prompt=system_prompt,
    tools=ToolBox(tools),
    attachments=attachments,
    hints=hints,
)
```

Define `AgentBundle` as a small `@dataclass` at the top of `agent_service.py`.

- [ ] **Step 3: `chat.py` builds `AgentContext` and runs the loop**

Replace the `AgentState.from_messages` + `ReactAgent` usage in `backend/api/v1/chat.py`:

```python
from agent.message import from_thread, keep_last_rounds, Message
from agent.context import AgentContext, RunVars
from common.chat.constants import DEFAULT_AGENT_HISTORY_ROUNDS

...
async with agent_service.create_agent(
    chat_request, tenant_id=tenant_id
) as bundle:
    msgs = from_thread(chat_request.messages)
    current_turn = msgs[-1]
    history = keep_last_rounds(msgs[:-1], DEFAULT_AGENT_HISTORY_ROUNDS)
    ctx = AgentContext(
        system_prompt=bundle.system_prompt,
        history=history,
        current_turn=current_turn,
        attachments=bundle.attachments,
        hints=bundle.hints,
        tools=bundle.tools,
        run_vars=RunVars(),
    )
    # input guardrail uses current_turn (clean user text) — unchanged logic
    async_response_gen = bundle.agent.run(ctx)
    response = await generate_reponse(
        chunk_gen=async_response_gen, ..., user_message=current_turn.to_wire()
    )
```

The `current_turn` Message is never mutated, so the history-save path needs no `deepcopy` and no system-time stripping — pass `current_turn.to_wire()` directly as the saved user message.

- [ ] **Step 4: Run the full backend test suite**

Run: `cd backend && python -m pytest ../tests/ -v`
Expected: PASS. Fix any import breakages surfaced by the signature changes (notably anything importing `AgentState`).

- [ ] **Step 5: Manual smoke test (attachment path)**

Per `docs/api/files_api.md`, upload a small text file and send a chat referencing it. Confirm the `[agent] model input:` log line shows the file body, and the answer uses it.

- [ ] **Step 6: Commit**

```bash
git add backend/service/agent/agent_service.py backend/api/v1/chat.py tests/
git commit -m "refactor(agent): wire chat.py + agent_service to Agent/AgentContext"
```

---

## Task 9: Delete dead code

**Files:**
- Delete: `backend/agent/react_agent.py`, `backend/agent/state.py`, `backend/agent/tool_utils.py`
- Modify: `backend/api/v1/chat.py` (remove `copy.deepcopy`), `backend/service/cache/session_history_manager.py` (drop system-time stripping if now unused)

- [ ] **Step 1: Confirm no remaining importers**

Run: `cd backend && grep -rn "react_agent\|agent.state\|AgentState\|tool_utils" --include="*.py" . | grep -v graphify`
Expected: no hits outside the files being deleted. Fix any that remain.

- [ ] **Step 2: Delete the files**

```bash
git rm backend/agent/react_agent.py backend/agent/state.py backend/agent/tool_utils.py
```

- [ ] **Step 3: Remove the now-unnecessary history hygiene**

In `chat.py`, revert `current_user_message = copy.deepcopy(...)` to `current_turn.to_wire()` (done in Task 8) and drop the unused `import copy`. In `session_history_manager.py`, the `[System Time:]` prefix is no longer written into the saved message (it's added in `build_messages` on the built copy), so `_clean_user_message`'s regex strip is dead — confirm via the existing test `tests/service/cache/test_session_history_manager.py` and simplify only if green.

- [ ] **Step 4: Run full suite**

Run: `cd backend && python -m pytest ../tests/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(agent): delete react_agent/state/tool_utils; drop deepcopy hygiene"
```

---

## Notes for the implementer

- `to_openai_tool(skip_length_check=True)`, `tool.async_fn`, and `parse_tool_arguments` already exist and are used by today's `react_agent`; reuse them as shown.
- The `ChoiceDeltaToolCall` reassembly across streamed chunks: the existing `llm.astream` already coalesces partial tool calls and emits them on a chunk's `tool_calls`; `_stream_turn` simply takes the last non-empty `chunk.tool_calls` (matching current behavior in `react_agent.run_async`). The `FakeLLM` test that splits a tool call across chunks (Risk in spec) should be added if `llm.astream` does NOT pre-coalesce — verify by reading `PaiLlm.astream` before Task 7.
- Keep loguru `[agent]` and `[attachments]` log tags — they are the debugging surface this refactor is meant to provide.
- **Tracing is a must-keep.** Before finishing Task 7, read `react_agent.run_async` lines 141–147 and apply the same instrumentation to `Agent.run`: decorate with `@pai_agent_wrapper` and wrap the body in `use_current_span(trace.get_current_span())` (imports: `from extensions.trace.pai_agent_wrapper import pai_agent_wrapper`, `from extensions.trace.base import use_current_span`, `from opentelemetry import trace`). Per-tool tracing already rides in `ToolBox._call_with_retry` via `instrument_async_call`. Add a smoke check that `agent.run` still emits a span.
- **`events.py` from the spec is intentionally omitted.** The plan uses the existing chunk types (`TextChunk`/`ReasoningChunk`/`ToolResultChunk`/`ErrorChunk` from `common.llm.models`) directly rather than re-exporting them under an `Event` alias — fewer indirections, and the SSE serializer already consumes those types. If a future reader wants the alias, it is a one-line `events.py`; not required here.
- **Verify `PaiLlm.astream` tool-call coalescing before Task 7.** If it does NOT merge partial `tool_calls` across chunks, `_stream_turn` must accumulate them by `index` (as a streaming OpenAI client does) instead of taking the last non-empty `chunk.tool_calls`; add a `FakeLLM` test that splits one tool call across two chunks to lock whichever behavior applies.
