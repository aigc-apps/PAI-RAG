# Agent Event Core + Chat Serializer (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Agent.run` emit a typed `AgentEvent` stream and re-point `/v1/chat/completions` at a serializer driven by those events — structurally fixing the dropped-usage, content-swallow, and invisible-timeout bugs.

**Architecture:** `Agent.run(ctx)` yields a discriminated `AgentEvent` union (text/reasoning/tool/usage/error). A `ChatCompletionsSerializer` maps that stream to `chat.completion.chunk` SSE (subsuming today's `convert_gen_to_stream_chat_completions`). The agent loop no longer yields OpenAI-shaped chunks; it speaks `AgentEvent`, adapting the LLM client's chunks at the model seam.

**Tech Stack:** Python 3.11, Pydantic v2, async generators, pytest, the existing `openai` SDK chunk types, loguru.

**Reference spec:** `docs/superpowers/specs/2026-06-26-agent-api-protocol-design.md` (this plan = migration steps 1–2). Phases 2–4 are separate plans.

**Scope note:** This phase keeps `agent/agent.py` in place (the `agent/core/` move is a later phase) and keeps the existing `ToolBox`. It only changes the *event type* the loop emits and the chat serializer that consumes it.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/agent/core/events.py` (new) | The `AgentEvent` discriminated union + `Usage` |
| `backend/agent/agent.py` (modify) | `run()`/`_stream_turn` emit `AgentEvent` instead of `TextChunk`/`ToolResultChunk`/`ErrorChunk` |
| `backend/api/protocol/chat_serializer.py` (new) | `AgentEvent` stream → `chat.completion.chunk` SSE strings (+ output guardrail + history save) |
| `backend/api/v1/chat.py` (modify) | Drive the new serializer over `Agent.run` |
| `backend/common/llm/utils.py` (modify) | Retire `convert_gen_to_stream_chat_completions`/`convert_gen_to_chat_completions` once chat.py is repointed |
| `tests/agent/test_events.py` (new) | `AgentEvent` unit tests |
| `tests/agent/test_agent_run.py` (modify) | Assert `Agent.run` emits the right event sequence |
| `tests/api/protocol/test_chat_serializer.py` (new) | Serializer golden + regression-lock tests |

---

## Task 1: `AgentEvent` union

**Files:**
- Create: `backend/agent/core/events.py`
- Test: `tests/agent/test_events.py`

- [ ] **Step 1: Write failing tests**

```text
# tests/agent/test_events.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from agent.core.events import (
    Usage, RunStarted, TextDelta, ReasoningDelta,
    ToolStarted, ToolCompleted, ToolResult, RunCompleted, RunFailed,
)


def test_each_event_has_stable_type_tag():
    assert TextDelta(text="hi").type == "text.delta"
    assert ReasoningDelta(text="r").type == "reasoning.delta"
    assert RunStarted(response_id="resp_1").type == "run.started"
    assert ToolStarted(call_id="c1", name="echo").type == "tool.started"
    assert ToolCompleted(call_id="c1", name="echo", arguments='{"x":1}').type == "tool.completed"
    assert ToolResult(call_id="c1", name="echo", ok=True, output="r").type == "tool.result"
    assert RunCompleted(usage=Usage(input=5, output=9, total=14), finish_reason="stop").type == "run.completed"
    assert RunFailed(message="timeout", error_type="llm_stream_timeout").type == "run.failed"


def test_tool_result_carries_error_when_not_ok():
    ev = ToolResult(call_id="c1", name="echo", ok=False, error="boom")
    assert ev.ok is False and ev.error == "boom" and ev.output is None


def test_usage_totals():
    u = Usage(input=5, output=9, total=14)
    assert (u.input, u.output, u.total) == (5, 9, 14)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest ../tests/agent/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.core'`

- [ ] **Step 3: Implement `backend/agent/core/events.py`**

Create `backend/agent/core/__init__.py` (empty) and:

```text
# backend/agent/core/events.py
from __future__ import annotations
from typing import Literal, Optional, Union
from pydantic import BaseModel


class Usage(BaseModel):
    input: int = 0
    output: int = 0
    total: int = 0


class RunStarted(BaseModel):
    type: Literal["run.started"] = "run.started"
    response_id: str
    conversation_id: Optional[str] = None


class TextDelta(BaseModel):
    type: Literal["text.delta"] = "text.delta"
    text: str


class ReasoningDelta(BaseModel):
    type: Literal["reasoning.delta"] = "reasoning.delta"
    text: str


class ToolStarted(BaseModel):
    type: Literal["tool.started"] = "tool.started"
    call_id: str
    name: str


class ToolCompleted(BaseModel):
    type: Literal["tool.completed"] = "tool.completed"
    call_id: str
    name: str
    arguments: str  # raw JSON string


class ToolResult(BaseModel):
    type: Literal["tool.result"] = "tool.result"
    call_id: str
    name: str
    ok: bool
    output: Optional[str] = None
    error: Optional[str] = None


class RunCompleted(BaseModel):
    type: Literal["run.completed"] = "run.completed"
    usage: Usage = Usage()
    finish_reason: str = "stop"


class RunFailed(BaseModel):
    type: Literal["run.failed"] = "run.failed"
    message: str
    error_type: str = "error"


AgentEvent = Union[
    RunStarted, TextDelta, ReasoningDelta, ToolStarted,
    ToolCompleted, ToolResult, RunCompleted, RunFailed,
]
```

- [ ] **Step 4: Run to verify pass** — `cd backend && python -m pytest ../tests/agent/test_events.py -v` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/agent/core/__init__.py backend/agent/core/events.py tests/agent/test_events.py
git commit -m "feat(agent): AgentEvent discriminated union"
```

---

## Task 2: `Agent.run` emits `AgentEvent`

Rewrite the event-producing parts of `backend/agent/agent.py` so the loop yields `AgentEvent` instead of `TextChunk`/`ReasoningChunk`/`ToolResultChunk`/`ErrorChunk`. The loop *control flow* (build_messages, budget.fit, `_stream_turn`, tool dispatch, return_direct, max-steps) is unchanged; only the yielded objects change. Read the current `backend/agent/agent.py` `_stream_turn` and `run` first.

**Files:**
- Modify: `backend/agent/agent.py`
- Test: `tests/agent/test_agent_run.py`

- [ ] **Step 1: Rewrite the test expectations to assert events**

Replace the body of `tests/agent/test_agent_run.py` helper `_collect` users so they assert on `AgentEvent`. Add imports and new tests (keep the `FakeLLM`/`_make_mock_tokenizer`/`_ctx`/`_box`/`_collect` helpers as-is; they already `await agent.run(ctx)`):

```text
# add near the other imports in tests/agent/test_agent_run.py
from agent.core.events import (
    TextDelta, ToolStarted, ToolCompleted, ToolResult, RunCompleted, RunFailed,
)
from openai.types.chat.chat_completion_chunk import CompletionUsage


@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_text_only_emits_textdeltas_then_completed(mock_get_tok):
    usage = CompletionUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
    llm = FakeLLM([[TextChunk(delta="he"), TextChunk(delta="llo"), TextChunk(delta="", usage=usage)]])
    agent = Agent(llm, max_steps=5)
    async def echo(x: str): return x
    out = _collect(agent, _ctx(_box(echo, "echo")))
    assert "".join(e.text for e in out if isinstance(e, TextDelta)) == "hello"
    completed = [e for e in out if isinstance(e, RunCompleted)]
    assert len(completed) == 1 and completed[0].usage.output == 2


@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_tool_call_emits_started_completed_result(mock_get_tok):
    llm = FakeLLM([
        [TextChunk(tool_calls=[tool_call(0, "c1", "echo", json.dumps({"x": "hi"}))])],
        [TextChunk(delta="done")],
    ])
    agent = Agent(llm, max_steps=5)
    async def echo(x: str): return f"echoed {x}"
    out = _collect(agent, _ctx(_box(echo, "echo")))
    types = [type(e).__name__ for e in out]
    assert "ToolStarted" in types and "ToolCompleted" in types and "ToolResult" in types
    tr = [e for e in out if isinstance(e, ToolResult)][0]
    assert tr.ok and "echoed hi" in tr.output and tr.call_id == "c1"
    assert "".join(e.text for e in out if isinstance(e, TextDelta)) == "done"


@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_idle_timeout_emits_run_failed(mock_get_tok, monkeypatch):
    import agent.agent as agent_mod
    monkeypatch.setattr(agent_mod, "LLM_STREAM_IDLE_TIMEOUT", 0)

    class HangingLLM:
        context_window = 110000
        max_tokens = 8000
        async def astream(self, messages, tools):
            async def gen():
                await asyncio.sleep(3600)
                yield TextChunk(delta="never")
            return gen()

    agent = Agent(HangingLLM(), max_steps=2)
    async def echo(x: str): return x
    out = _collect(agent, _ctx(_box(echo, "echo")))
    failed = [e for e in out if isinstance(e, RunFailed)]
    assert len(failed) == 1 and failed[0].error_type == "llm_stream_timeout"
```

Delete the now-obsolete chunk-asserting tests in that file (`test_plain_text_answer_streams_and_stops`, `test_tool_call_then_final_answer`, `test_return_direct_short_circuits`, `test_max_steps_emits_notice`, `test_idle_timeout_yields_error_chunk`, `test_usage_only_terminal_chunk_is_forwarded`) — they assert the old `TextChunk` contract. Replace `test_return_direct_short_circuits` and `test_max_steps_emits_notice` with event versions:

```text
@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_return_direct_emits_textdelta_then_completed(mock_get_tok):
    llm = FakeLLM([[TextChunk(tool_calls=[tool_call(0, "c1", "faq", "{}")])]])
    agent = Agent(llm, max_steps=5)
    async def faq(): return json.dumps({"result": [{"content": "FAQ answer"}]})
    out = _collect(agent, _ctx(_box(faq, "faq", return_direct=True)))
    assert any(isinstance(e, TextDelta) and "FAQ answer" in e.text for e in out)
    assert any(isinstance(e, RunCompleted) for e in out)


@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_max_steps_emits_completed_with_incomplete_reason(mock_get_tok):
    turns = [[TextChunk(tool_calls=[tool_call(0, f"c{i}", "echo", "{}")])] for i in range(3)]
    agent = Agent(FakeLLM(turns), max_steps=2)
    async def echo(): return "x"
    out = _collect(agent, _ctx(_box(echo, "echo")))
    completed = [e for e in out if isinstance(e, RunCompleted)]
    assert len(completed) == 1 and completed[0].finish_reason == "max_steps"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && python -m pytest ../tests/agent/test_agent_run.py -v`
Expected: FAIL — events not emitted / import errors (the loop still yields `TextChunk`).

- [ ] **Step 3: Rewrite the emission in `backend/agent/agent.py`**

Add imports at top: `from agent.core.events import (RunStarted, TextDelta, ReasoningDelta, ToolStarted, ToolCompleted, ToolResult, RunCompleted, RunFailed, Usage)`.

Change `_stream_turn` to emit `TextDelta`/`ReasoningDelta` and accumulate a `Usage` (read it off the usage-bearing chunk), storing usage in the sink:

```text
    async def _stream_turn(self, messages, tools, sink):
        """Stream one model turn as AgentEvents. sink["last"]=(text, tool_calls);
        sink["usage"] accumulates; on ErrorChunk/timeout sink["error"] is set."""
        wire = [m.to_wire() for m in messages]
        stream = await self.llm.astream(messages=wire, tools=tools.openai_schema() if tools else [])
        text, tool_calls = "", []
        try:
            async for chunk in _iter_with_idle_timeout(stream, LLM_STREAM_IDLE_TIMEOUT):
                if isinstance(chunk, ErrorChunk):
                    sink["error"] = RunFailed(message=chunk.error_message or chunk.delta or "LLM error",
                                              error_type=chunk.error_type or "llm")
                    sink["last"] = (text, tool_calls)
                    return
                if chunk.tool_calls:
                    tool_calls = chunk.tool_calls
                if chunk.usage:
                    sink["usage"] = Usage(input=chunk.usage.prompt_tokens or 0,
                                          output=chunk.usage.completion_tokens or 0,
                                          total=chunk.usage.total_tokens or 0)
                if isinstance(chunk, ReasoningChunk) and chunk.reasoning_delta:
                    yield ReasoningDelta(text=chunk.reasoning_delta)
                elif chunk.delta:
                    text += chunk.delta
                    yield TextDelta(text=chunk.delta)
        except asyncio.TimeoutError:
            logger.error(f"LLM stream idle >{LLM_STREAM_IDLE_TIMEOUT}s; aborting.")
            sink["error"] = RunFailed(
                message=f"模型调用超时：{LLM_STREAM_IDLE_TIMEOUT}s 内未收到任何响应分片。",
                error_type="llm_stream_timeout")
            sink["last"] = (text, tool_calls)
            return
        sink["last"] = (text, tool_calls)
```

Rewrite the loop body in `run`'s inner `gen()` to emit events (replace the chunk yields). Keep the existing `pairs` construction and dispatch; change only what is yielded and the terminal events:

```text
        # inside gen(), before the loop:
        yield RunStarted(response_id=getattr(ctx, "response_id", "") or "resp_local")
        usage = Usage()
        for _step in range(self.max_steps):
            messages = self.budget.fit(messages)
            sink = {"last": ("", []), "error": None, "usage": None}
            async for ev in self._stream_turn(messages, ctx.tools, sink):
                yield ev
            if sink["usage"]:
                usage = sink["usage"]
            if sink["error"] is not None:
                yield sink["error"]            # RunFailed
                return
            text, raw_tcs = sink["last"]

            if not raw_tcs:
                if text:
                    messages.append(Message("assistant", text))
                yield RunCompleted(usage=usage, finish_reason="stop")
                return

            pairs = [(raw, ToolCall(id=raw.id, name=raw.function.name, arguments=raw.function.arguments or ""))
                     for raw in raw_tcs if raw.type == "function" and ctx.tools.get(raw.function.name)]
            if not pairs:
                bad = raw_tcs[0]
                messages.append(Message("assistant", text or None,
                    tool_calls=[ToolCall(bad.id, bad.function.name, bad.function.arguments or "")]))
                messages.append(Message("tool", content=f"Error: tool '{bad.function.name}' not available.", tool_call_id=bad.id))
                continue

            for raw, tc in pairs:
                yield ToolStarted(call_id=tc.id, name=tc.name)
                yield ToolCompleted(call_id=tc.id, name=tc.name, arguments=tc.arguments)
            for idx, (raw, tc) in enumerate(pairs):
                result = await ctx.tools.dispatch(tc)
                messages.append(Message("assistant", text if idx == 0 else None, tool_calls=[tc]))
                capped = self.budget.cap_tool_result(result.message.content) if result.message.content else result.message.content
                messages.append(Message("tool", content=capped, tool_call_id=tc.id))
                yield ToolResult(call_id=tc.id, name=tc.name, ok=result.ok,
                                 output=result.content, error=result.error)
                if ctx.tools.is_return_direct(tc.name) and result.ok:
                    direct = _format_return_direct(result.content)
                    if direct:
                        yield TextDelta(text=direct)
                    yield RunCompleted(usage=usage, finish_reason="stop")
                    return
        yield RunCompleted(usage=usage, finish_reason="max_steps")
```

Remove the now-unused imports of `TextChunk`/`ToolResultChunk` from agent.py if nothing else uses them (keep `ErrorChunk`/`ReasoningChunk` — still consumed from the LLM client; keep `TextChunk` only if still referenced). Run `grep -n "TextChunk\|ToolResultChunk" backend/agent/agent.py` and drop dead imports.

- [ ] **Step 4: Run to verify pass** — `cd backend && python -m pytest ../tests/agent/test_agent_run.py ../tests/agent/ -v` → all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/agent/agent.py tests/agent/test_agent_run.py
git commit -m "feat(agent): run() emits AgentEvent stream (text/reasoning/tool/usage/error)"
```

---

## Task 3: `ChatCompletionsSerializer`

A new serializer: `AgentEvent` stream → `chat.completion.chunk` SSE JSON strings. It subsumes `convert_gen_to_stream_chat_completions`. Read that function in `backend/common/llm/utils.py` to port the `ChatCompletionChunk` construction, the output-guardrail hook, and the history-save block — but drive them off typed events. This task implements the **stream** serializer; the sync path (`convert_gen_to_chat_completions`) is ported the same way in Step 6.

**Files:**
- Create: `backend/api/protocol/__init__.py`, `backend/api/protocol/chat_serializer.py`
- Test: `tests/api/protocol/test_chat_serializer.py`

- [ ] **Step 1: Write failing tests (regression locks first)**

```text
# tests/api/protocol/test_chat_serializer.py
import sys, os, json, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))
from agent.core.events import TextDelta, RunCompleted, RunFailed, Usage, ToolResult
from api.protocol.chat_serializer import serialize_chat_stream


async def _events(*evs):
    for e in evs:
        yield e


def _collect(gen):
    async def run():
        return [json.loads(s) async for s in gen]
    return asyncio.run(run())


def test_text_deltas_become_content_chunks():
    out = _collect(serialize_chat_stream(_events(
        TextDelta(text="he"), TextDelta(text="llo"),
        RunCompleted(usage=Usage(input=5, output=2, total=7))), model="m"))
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in out)
    assert "hello" in content


def test_usage_reaches_final_chunk():  # regression: usage must not be dropped
    out = _collect(serialize_chat_stream(_events(
        TextDelta(text="hi"),
        RunCompleted(usage=Usage(input=5, output=9, total=14))), model="m"))
    stop = [c for c in out if c["choices"][0].get("finish_reason") == "stop"][0]
    assert stop["usage"]["completion_tokens"] == 9 and stop["usage"]["total_tokens"] == 14


def test_run_failed_message_is_visible():  # regression: invisible-timeout bug
    out = _collect(serialize_chat_stream(_events(
        RunFailed(message="模型调用超时", error_type="llm_stream_timeout")), model="m"))
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in out)
    assert "模型调用超时" in text
```

- [ ] **Step 2: Run to verify failure** — `cd backend && python -m pytest ../tests/api/protocol/test_chat_serializer.py -v` → `ModuleNotFoundError: api.protocol.chat_serializer`.

- [ ] **Step 3: Implement `backend/api/protocol/chat_serializer.py`**

Create `backend/api/protocol/__init__.py` (empty) and `tests/api/protocol/__init__.py` (empty). Then:

```text
# backend/api/protocol/chat_serializer.py
from __future__ import annotations
import json, time, uuid
from typing import AsyncIterator
from agent.core.events import (
    AgentEvent, TextDelta, ReasoningDelta, ToolStarted, ToolCompleted,
    ToolResult, RunCompleted, RunFailed,
)


def _chunk(chat_id, model, *, content=None, reasoning=None, finish_reason=None,
           usage=None, extra=None) -> str:
    delta = {"role": "assistant"}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    body = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        body["usage"] = usage
    if extra:
        body.update(extra)
    return json.dumps(body, ensure_ascii=False)


async def serialize_chat_stream(events: AsyncIterator[AgentEvent], *, model: str) -> AsyncIterator[str]:
    """AgentEvent stream -> chat.completion.chunk JSON strings (one per yield).
    The caller wraps each with the SSE 'data: ' prefix and trailing [DONE]."""
    chat_id = "chatcmpl-" + uuid.uuid4().hex
    async for ev in events:
        if isinstance(ev, TextDelta):
            yield _chunk(chat_id, model, content=ev.text)
        elif isinstance(ev, ReasoningDelta):
            yield _chunk(chat_id, model, reasoning=ev.text)
        elif isinstance(ev, ToolStarted):
            # PAI-RAG tool visibility (custom field; ignored by vanilla OpenAI clients)
            yield _chunk(chat_id, model, content="", extra={"actions": [{"id": ev.call_id, "name": ev.name}]})
        elif isinstance(ev, ToolResult):
            yield _chunk(chat_id, model, content="",
                         extra={"observation": {"call_id": ev.call_id, "ok": ev.ok,
                                                 "output": ev.output, "error": ev.error}})
        elif isinstance(ev, RunFailed):
            # message MUST be visible content (the invisible-timeout fix)
            yield _chunk(chat_id, model, content=ev.message, extra={"error_type": ev.error_type})
            yield _chunk(chat_id, model, finish_reason="stop")
            return
        elif isinstance(ev, RunCompleted):
            yield _chunk(chat_id, model, finish_reason="stop", usage={
                "prompt_tokens": ev.usage.input,
                "completion_tokens": ev.usage.output,
                "total_tokens": ev.usage.total,
            })
            return
        # ToolCompleted: no chat.completion.chunk representation; skipped
```

- [ ] **Step 4: Run to verify pass** — `cd backend && python -m pytest ../tests/api/protocol/test_chat_serializer.py -v` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/api/protocol/__init__.py backend/api/protocol/chat_serializer.py tests/api/protocol/
git commit -m "feat(api): ChatCompletionsSerializer over AgentEvent (usage + error regression locks)"
```

- [ ] **Step 6: Port output-guardrail + history-save**

Read `convert_gen_to_stream_chat_completions` in `backend/common/llm/utils.py` lines 100–256. The two responsibilities beyond chunk-shaping are (a) the streaming output-guardrail check (`checker.acheck_output`, `output_check_result.reject`) and (b) the session-history save (`final_content` accumulation + `session_history_manager.save_messages`). Add an outer wrapper in `chat_serializer.py` that consumes the `AgentEvent` stream once, accumulates `final_content` from `TextDelta`/`RunFailed`, collects tool history from `ToolResult`, runs the same guardrail logic, and saves history — delegating chunk formatting to `serialize_chat_stream`. Mirror the existing signature (`session`, `user_id`, `session_id`, `user_message`, `enable_output_check`, `checker`, `guardrail_hint`) so `chat.py` can swap it in. Add a test that a `RunFailed` still triggers the history-save path with the error text and that a rejecting guardrail yields the advice chunk. Commit:

```bash
git add backend/api/protocol/chat_serializer.py tests/api/protocol/test_chat_serializer.py
git commit -m "feat(api): chat serializer output-guardrail + history-save parity"
```

---

## Task 4: Re-point `/v1/chat/completions`

**Files:**
- Modify: `backend/api/v1/chat.py`
- Modify: `backend/common/llm/utils.py` (retire the old serializers once unused)

- [ ] **Step 1: Switch `chat.py` to the new serializer**

In `backend/api/v1/chat.py`, `generate_reponse` currently calls `convert_gen_to_stream_chat_completions(chunk_gen=...)` / `convert_gen_to_chat_completions(...)` over the agent's chunk stream. Change it to consume `Agent.run`'s `AgentEvent` stream via the new `serialize_chat_stream` wrapper (the guardrail+history variant from Task 3 Step 6), wrapping each yielded JSON string as `f"data: {s}\n\n"` and ending with `data: [DONE]\n\n` for SSE (match the existing SSE framing in `EventSourceResponse`/`generate_reponse`). Keep the non-stream branch using the ported sync serializer.

- [ ] **Step 2: Run the agent + serializer suites + a chat import smoke**

Run: `cd backend && python -c "import api.v1.chat" && python -m pytest ../tests/agent/ ../tests/api/protocol/ -q`
Expected: imports clean; all pass.

- [ ] **Step 3: Retire the old serializer**

Once `chat.py` no longer imports them, delete `convert_gen_to_stream_chat_completions` and `convert_gen_to_chat_completions` from `backend/common/llm/utils.py` (and any now-unused chunk imports). Confirm: `grep -rn "convert_gen_to_stream_chat_completions\|convert_gen_to_chat_completions" backend --include="*.py" | grep -v graphify` → no live references.

- [ ] **Step 4: Full check**

Run: `cd backend && python -m pytest ../tests/agent/ ../tests/api/protocol/ -q`
Expected: all pass.

- [ ] **Step 5: Manual smoke (configured env)**

Send a `stream:true` chat (a greeting, and one that triggers a tool). Confirm: text streams, the final chunk carries non-zero `usage`, and an induced LLM timeout shows the timeout message (not an empty response).

- [ ] **Step 6: Commit**

```bash
git add backend/api/v1/chat.py backend/common/llm/utils.py
git commit -m "refactor(api): drive /v1/chat/completions from AgentEvent serializer; retire legacy chunk serializers"
```

---

## Notes for the implementer

- The earlier band-aid fix (`fix(agent): forward usage-only terminal chunk`) is **superseded** here: usage now rides `RunCompleted`, so the `elif chunk.delta or chunk.usage` guard in `_stream_turn` is replaced by explicit `if chunk.usage: sink["usage"]=...`. Make sure no path still depends on the old guard.
- `@pai_agent_wrapper` on `run` and the inner `gen()`/`use_current_span` structure stay exactly as-is — only the yielded objects change. Tracing input capture (added earlier for `AgentContext`) is unaffected.
- `ctx.response_id` may not exist yet (it arrives with the Responses endpoint in a later phase). `RunStarted(response_id=getattr(ctx, "response_id", "") or "resp_local")` keeps Phase 1 self-contained.
- Do NOT introduce `agent/core/agent.py` or the clean `Tool` abstraction here — those are later phases. Keep `agent.py` and `ToolBox` in place.
- **Green-per-task / ordering:** after Task 2 the live `/v1/chat/completions` endpoint is transiently inconsistent (the agent emits `AgentEvent`, but `chat.py` still feeds the old chunk serializer) until Task 4 repoints it. The **automated suites stay green throughout** (the endpoint has no automated test in this phase; agent + serializer unit tests cover the changed code). So: do **not** run a live chat smoke between Task 2 and Task 4, and treat "all unit tests pass" as the per-task bar. If you want the live endpoint working between tasks, add a throwaway `AgentEvent → TextChunk` adapter in `chat.py` and delete it in Task 4 — optional, not required for the TDD flow.
- The `actions`/`observation` custom fields emitted for `ToolStarted`/`ToolResult` are PAI-RAG additions that vanilla OpenAI clients ignore; the frontend's existing adapter already reads them. Strict-OpenAI behavior is a `/v1/responses` concern (later phase), not this shim.
