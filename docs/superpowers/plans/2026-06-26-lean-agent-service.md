# Lean Agent Service — LLM + builder + Responses serializer + endpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the lean `backend/app/` service actually run — a clean `LeanLLM` streaming client, a request→`AgentContext` builder over the store, an `AgentEvent`→OpenAI-Responses serializer (sync + stream), `/v1/responses` (+ a `/v1/chat/completions` shim), and a minimal `main.py` that boots with zero RAG/llamaindex/tokenizer/trace deps.

**Architecture:** Builds on the foundation (Tasks done in `2026-06-26-lean-agent-foundation.md`: lean agent boot, schema, `ResponseStore`/`InMemoryStore`/`SqlStore`). The agent core (`agent/agent.py` `Agent.run(ctx) -> AsyncIterator[AgentEvent]`) and the existing `api/protocol/chat_serializer.py` are reused as-is. New code: `app/llm.py`, `app/schemas.py`, `app/builder.py`, `api/protocol/responses_serializer.py`, `app/routes/`, `app/main.py`. The agent runs the full tool loop server-side within one `run`; the serializer turns its events into OpenAI Responses wire output **and** a parallel list of store-ready item dicts the route persists.

**Tech Stack:** Python 3.11, FastAPI + Starlette `EventSourceResponse`/`StreamingResponse`, `openai` 2.30.0 (`AsyncOpenAI` + `openai.types.responses`), SQLModel, pydantic, pytest. Tests run from `backend/` with paths relative to it (`cd backend && python -m pytest ../tests/app/... -q`).

**Reference spec:** `docs/superpowers/specs/2026-06-26-standalone-lean-agent-design.md` (steps 4–6). Branch: `personal/yfei/agent-core`. Frontend migration to `/v1/responses` is a SEPARATE later plan.

---

## Key existing contracts (verified, do not re-derive)

- **`agent/agent.py`** — `class Agent` with `async def run(self, ctx: AgentContext) -> AsyncIterator[AgentEvent]` (returns the inner `gen()` async generator). It calls `self.llm.astream(messages=wire, tools=...)` inside `_stream_turn`. The `Agent.__init__` takes an `llm` (anything with `astream`) plus the message manager; READ `agent/agent.py` `__init__` for the exact constructor params before wiring in Task 5/6.
- **LLM chunk contract** (`common/llm/models.py`): `TextChunk(delta:str="", tool_calls:List[ChoiceDeltaToolCall]=[], usage:Optional[CompletionUsage]=None, stage:str="", trace_id:str="")`; `ReasoningChunk(TextChunk)` adds `reasoning_delta:str=""`; `ErrorChunk(TextChunk)` adds `error_message:str="", exception:str|None=None, error_type:str=""`. `_stream_turn` reads `chunk.tool_calls`, `chunk.usage` (`.prompt_tokens/.completion_tokens/.total_tokens`), `chunk.reasoning_delta`, `chunk.delta`, and `isinstance(chunk, ErrorChunk)`.
- **Tool-call coalescing helper**: `common/llm/llm_model.py` `update_tool_calls(tool_calls, tool_calls_delta)` — reuse it.
- **`AgentEvent`** (`agent/core/events.py`): `Usage(input,output,total)`, `RunStarted(response_id, conversation_id=None)`, `TextDelta(text)`, `ReasoningDelta(text)`, `ToolStarted(call_id, name)`, `ToolCompleted(call_id, name, arguments)`, `ToolResult(call_id, name, ok, output=None, error=None)`, `RunCompleted(usage:Usage, finish_reason="stop")`, `RunFailed(message, error_type="error")`.
- **`AgentContext`** (`agent/context.py`): dataclass `system_prompt:str, history:List[Message], current_turn:Message, attachments:List, hints:List[str], tools:object (ToolBox), run_vars:RunVars`. `RunVars()` default-constructs.
- **`Message`** (`agent/message.py`): dataclass `role:str, content=None, tool_calls:Optional[List[ToolCall]]=None, tool_call_id:Optional[str]=None`; `ToolCall(id, name, arguments)`. `from_thread(raw:List[dict])->List[Message]` normalizes OpenAI-style dicts.
- **`Tool`/`ToolBox`** (`agent/tools/base.py`): `ToolBox(tools:List[Tool])`; `.openai_schema()->List[dict]`. Empty `ToolBox([])` for the lean default (no tools yet).
- **`openai.types.responses`** (SDK 2.30.0) verified field shapes:
  - `Response(id:str, created_at:float, model:str, object="response", output:list, parallel_tool_calls:bool, tool_choice, tools:list, status, usage?, error?, previous_response_id?, metadata?)`.
  - `ResponseOutputMessage(id, content:List[ResponseOutputText|...], role="assistant", status, type="message")`; `ResponseOutputText(annotations=[], text, type="output_text")`.
  - `ResponseFunctionToolCall(arguments:str, call_id:str, name:str, type="function_call", id?, status?)`.
  - `ResponseReasoningItem(id, summary:List[Summary], type="reasoning", content?:List[Content], status?)`; `Summary(text, type="summary_text")`, `Content(text, type="reasoning_text")`.
  - `ResponseUsage(input_tokens:int, input_tokens_details:InputTokensDetails(cached_tokens:int), output_tokens:int, output_tokens_details:OutputTokensDetails(reasoning_tokens:int), total_tokens:int)`.
  - `ResponseError(code:Literal[...], message:str)` — `code` is a constrained literal; **always use `code="server_error"`** for agent errors.
  - Streaming events all carry `sequence_number:int` and a `type` literal. Used: `ResponseCreatedEvent(response, sequence_number, type="response.created")`, `ResponseInProgressEvent(...)`, `ResponseCompletedEvent(...)`, `ResponseFailedEvent(...)`, `ResponseOutputItemAddedEvent(item, output_index, sequence_number, type)`, `ResponseOutputItemDoneEvent(...)`, `ResponseContentPartAddedEvent(content_index, item_id, output_index, part, sequence_number, type)`, `ResponseContentPartDoneEvent(...)`, `ResponseTextDeltaEvent(content_index, delta, item_id, logprobs=[], output_index, sequence_number, type="response.output_text.delta")`, `ResponseTextDoneEvent(content_index, item_id, logprobs=[], output_index, sequence_number, text, type="response.output_text.done")`, `ResponseFunctionCallArgumentsDeltaEvent(delta, item_id, output_index, sequence_number, type)`, `ResponseFunctionCallArgumentsDoneEvent(arguments, item_id, name, output_index, sequence_number, type)`, `ResponseReasoningTextDeltaEvent(content_index, delta, item_id, output_index, sequence_number, type="response.reasoning_text.delta")`.
  - Every event serializes to SSE via `event.model_dump_json()`. Parse-back in tests via `openai.types.responses.ResponseStreamEvent` `TypeAdapter`.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/app/llm.py` (new) | `LeanLLM.astream()` — clean streaming wrapper over `AsyncOpenAI` emitting the chunk contract |
| `backend/app/schemas.py` (new) | `ResponsesRequest` pydantic model (OpenAI Responses request, lenient `extra="ignore"`) |
| `backend/app/builder.py` (new) | `items_to_messages()`, `build_context(request, store)` → `(AgentContext, conversation_id)` |
| `backend/api/protocol/responses_serializer.py` (new) | `AgentEvent` → OpenAI Responses: `serialize_response_sync` (+ `serialize_response_stream`) returning a `Response` and store-ready item dicts |
| `backend/app/routes/__init__.py` (new) | package marker |
| `backend/app/routes/responses.py` (new) | `POST /v1/responses` (stream+sync), `GET`, `DELETE`; linking + persistence |
| `backend/app/routes/chat.py` (new) | `POST /v1/chat/completions` shim over the existing `chat_serializer` |
| `backend/app/deps.py` (new) | app state accessors: `get_store()`, `get_llm()`, `get_agent()` (built from settings) |
| `backend/app/main.py` (new) | FastAPI app; lifespan `create_all`; mount routes; wire store + LLM from `Settings` |
| `tests/app/test_*` (new) | unit + integration tests |

---

## Task 1: `LeanLLM` streaming client

Mirror `PaiLlm.astream`'s chunk-emission contract with a minimal clean client. READ `backend/common/llm/llm_model.py` (`PaiLlm.astream` + `update_tool_calls`) and `backend/common/llm/models.py` first.

**Files:** Create `backend/app/llm.py`. Test: `tests/app/test_llm.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/app/test_llm.py
import sys, os, asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.llm import LeanLLM
from common.llm.models import TextChunk, ErrorChunk


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None, reasoning_content=None):
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class _FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class _FakeChunk:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


class _FakeUsage:
    prompt_tokens = 3
    completion_tokens = 5
    total_tokens = 8


class _FakeStream:
    """Async-iterable stand-in for the openai streaming response."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c

        return gen()


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks

    async def create(self, **kwargs):
        return _FakeStream(self._chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = type("C", (), {"completions": _FakeCompletions(chunks)})()


def test_astream_emits_text_and_usage():
    async def run():
        chunks = [
            _FakeChunk([_FakeChoice(_FakeDelta(content="Hell"))]),
            _FakeChunk([_FakeChoice(_FakeDelta(content="o"))]),
            _FakeChunk([_FakeChoice(_FakeDelta())], usage=_FakeUsage()),
        ]
        llm = LeanLLM(base_url="x", api_key="x", model="m")
        llm.client = _FakeClient(chunks)
        out = [
            c
            async for c in llm.astream(
                messages=[{"role": "user", "content": "hi"}], tools=[]
            )
        ]
        text = "".join(c.delta for c in out)
        assert text == "Hello"
        assert any(c.usage and c.usage.total_tokens == 8 for c in out)

    asyncio.run(run())


def test_astream_coalesces_tool_calls():
    async def run():
        class _TC:
            def __init__(self, index, id=None, name=None, args=None):
                self.index = index
                self.id = id
                self.type = "function"
                self.function = type(
                    "F", (), {"name": name, "arguments": args}
                )()

        chunks = [
            _FakeChunk(
                [
                    _FakeChoice(
                        _FakeDelta(
                            tool_calls=[
                                _TC(0, id="call_1", name="get", args='{"a"')
                            ]
                        )
                    )
                ]
            ),
            _FakeChunk(
                [_FakeChoice(_FakeDelta(tool_calls=[_TC(0, args=":1}")]))]
            ),
            _FakeChunk([_FakeChoice(_FakeDelta())], usage=_FakeUsage()),
        ]
        llm = LeanLLM(base_url="x", api_key="x", model="m")
        llm.client = _FakeClient(chunks)
        out = [c async for c in llm.astream(messages=[], tools=[])]
        final_calls = [c.tool_calls for c in out if c.tool_calls][-1]
        assert final_calls[0].function.name == "get"
        assert final_calls[0].function.arguments == '{"a":1}'

    asyncio.run(run())


def test_astream_error_yields_error_chunk():
    async def run():
        class _BoomCompletions:
            async def create(self, **kwargs):
                raise RuntimeError("boom")

        llm = LeanLLM(base_url="x", api_key="x", model="m")
        llm.client = type(
            "C2",
            (),
            {"chat": type("C", (), {"completions": _BoomCompletions()})()},
        )()
        out = [c async for c in llm.astream(messages=[], tools=[])]
        assert len(out) == 1 and isinstance(out[0], ErrorChunk)
        assert out[0].error_type == "llm"

    asyncio.run(run())
```

- [ ] **Step 2: Run to verify failure** — `cd backend && python -m pytest ../tests/app/test_llm.py -v` → `ModuleNotFoundError: app.llm`.

- [ ] **Step 3: Implement `backend/app/llm.py`**

```python
from __future__ import annotations
import traceback
from typing import List, Optional
from openai import AsyncOpenAI
from loguru import logger
from common.llm.models import TextChunk, ReasoningChunk, ErrorChunk
from common.llm.llm_model import update_tool_calls

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 2


class LeanLLM:
    """Minimal streaming client over openai.AsyncOpenAI emitting the agent's chunk
    contract (TextChunk / ReasoningChunk / ErrorChunk). No model registry, no dashscope.

    Its only contract is what `Agent._stream_turn` consumes: `astream(messages, tools)`
    yielding chunks with `.delta` / `.tool_calls` / `.usage` (+ `.reasoning_delta` on
    ReasoningChunk). Tool-call deltas are coalesced by index via `update_tool_calls`.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        enable_thinking: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def astream(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs,
    ):
        tool_calls = []
        tools_to_use = tools or None
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tools_to_use,
                stream_options={"include_usage": True},
                **kwargs,
            )
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                choices = getattr(chunk, "choices", None) or []
                delta_obj = choices[0].delta if choices else None
                if delta_obj is not None and getattr(
                    delta_obj, "tool_calls", None
                ):
                    tool_calls = update_tool_calls(
                        tool_calls, delta_obj.tool_calls
                    )
                content = (
                    getattr(delta_obj, "content", None) or ""
                    if delta_obj
                    else ""
                )
                reasoning = (
                    (getattr(delta_obj, "reasoning_content", None) or "")
                    if delta_obj
                    else ""
                )
                if reasoning:
                    yield ReasoningChunk(
                        delta="",
                        reasoning_delta=reasoning,
                        tool_calls=tool_calls,
                        usage=usage,
                    )
                elif content or tool_calls or usage is not None:
                    yield TextChunk(
                        delta=content, tool_calls=tool_calls, usage=usage
                    )
        except Exception as ex:
            logger.error(f"LeanLLM stream error: {traceback.format_exc()}")
            yield ErrorChunk(
                delta=f"{ex}",
                error_message=str(ex),
                exception=str(ex),
                error_type="llm",
            )
```

- [ ] **Step 4: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_llm.py -v` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/llm.py tests/app/test_llm.py
git commit -m "feat(app): LeanLLM streaming client (chunk contract over AsyncOpenAI)"
```

---

## Task 2: Responses request schema + `build_context`

`ResponsesRequest` (lenient: ignore unknown fields like `enable_agent`/`kb_ids`), plus a builder that resolves prior history from the store and assembles the `AgentContext`. READ `backend/app/store/base.py` (Item/Conversation/StoredResponse + `ResponseStore`), `backend/agent/context.py`, `backend/agent/message.py`.

**Files:** Create `backend/app/schemas.py`, `backend/app/builder.py`. Test: `tests/app/test_builder.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/app/test_builder.py
import sys, os, asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.schemas import ResponsesRequest
from app.builder import build_context, items_to_messages
from app.store.memory import InMemoryStore
from app.store.base import Item, StoredResponse


def test_request_ignores_unknown_fields_and_parses_input():
    req = ResponsesRequest(
        model="m", input="hello", enable_agent=True, kb_ids=["k1"]
    )
    assert req.input == "hello" and req.model == "m"


def test_build_context_from_string_input():
    async def run():
        st = InMemoryStore()
        req = ResponsesRequest(
            model="m", input="hi there", instructions="be terse"
        )
        ctx, conv_id = await build_context(req, st)
        assert ctx.current_turn.role == "user"
        assert ctx.current_turn.content == "hi there"
        assert ctx.system_prompt == "be terse"
        assert ctx.history == []
        assert conv_id is not None

    asyncio.run(run())


def test_build_context_resolves_previous_response_history():
    async def run():
        st = InMemoryStore()
        conv = await st.create_conversation()
        await st.append_items(
            conv.id,
            [
                Item(
                    type="message",
                    role="user",
                    content={"text": "q1"},
                    response_id="resp_1",
                ),
                Item(
                    type="message",
                    role="assistant",
                    content={"text": "a1"},
                    response_id="resp_1",
                ),
            ],
        )
        await st.save_response(
            StoredResponse(
                id="resp_1",
                conversation_id=conv.id,
                model="m",
                status="completed",
            )
        )
        req = ResponsesRequest(
            model="m", input="q2", previous_response_id="resp_1"
        )
        ctx, conv_id = await build_context(req, st)
        assert conv_id == conv.id
        assert [m.role for m in ctx.history] == ["user", "assistant"]
        assert ctx.history[0].content == "q1"
        assert ctx.current_turn.content == "q2"

    asyncio.run(run())


def test_items_to_messages_handles_function_call_and_output():
    msgs = items_to_messages(
        [
            Item(
                type="function_call",
                content={"call_id": "c1", "name": "get", "arguments": "{}"},
            ),
            Item(
                type="function_call_output",
                content={"call_id": "c1", "output": "42"},
            ),
        ]
    )
    assert msgs[0].role == "assistant" and msgs[0].tool_calls[0].id == "c1"
    assert (
        msgs[1].role == "tool"
        and msgs[1].tool_call_id == "c1"
        and msgs[1].content == "42"
    )


def test_build_context_conflicting_ids_raises_value_error():
    async def run():
        st = InMemoryStore()
        c1 = await st.create_conversation()
        c2 = await st.create_conversation()
        await st.save_response(
            StoredResponse(
                id="resp_x",
                conversation_id=c1.id,
                model="m",
                status="completed",
            )
        )
        req = ResponsesRequest(
            model="m",
            input="q",
            previous_response_id="resp_x",
            conversation=c2.id,
        )
        import pytest

        with pytest.raises(ValueError):
            await build_context(req, st)

    asyncio.run(run())
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: app.schemas`.

- [ ] **Step 3: Implement `backend/app/schemas.py`**

```python
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict


class ResponsesRequest(BaseModel):
    # Lenient: the current frontend sends extra fields (enable_agent, kb_ids, ...).
    # Accept and ignore them for now; tools are wired in a later plan.
    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = None
    input: Union[str, List[Dict[str, Any]]] = ""
    instructions: Optional[str] = None
    previous_response_id: Optional[str] = None
    conversation: Optional[str] = None
    store: bool = True
    stream: bool = False
    metadata: Optional[Dict[str, str]] = None
    tools: Optional[List[Dict[str, Any]]] = None
```

- [ ] **Step 4: Implement `backend/app/builder.py`**

```python
from __future__ import annotations
from typing import List, Optional, Tuple
from agent.context import AgentContext, RunVars
from agent.message import Message, ToolCall
from agent.tools.base import ToolBox
from app.schemas import ResponsesRequest
from app.store.base import Item

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _item_text(content: dict) -> str:
    if "text" in content:
        return content["text"] or ""
    # tolerate OpenAI-style message content arrays
    parts = content.get("content")
    if isinstance(parts, list):
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    if isinstance(parts, str):
        return parts
    return ""


def items_to_messages(items: List[Item]) -> List[Message]:
    """Convert stored conversation items (history source of truth) into agent Messages.
    Reasoning items are skipped (not replayed to the model)."""
    msgs: List[Message] = []
    for it in items:
        if it.type == "message":
            msgs.append(
                Message(role=it.role or "user", content=_item_text(it.content))
            )
        elif it.type == "function_call":
            c = it.content
            msgs.append(
                Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id=c.get("call_id", ""),
                            name=c.get("name", ""),
                            arguments=c.get("arguments", "") or "",
                        )
                    ],
                )
            )
        elif it.type == "function_call_output":
            c = it.content
            msgs.append(
                Message(
                    role="tool",
                    tool_call_id=c.get("call_id", ""),
                    content=c.get("output", ""),
                )
            )
        # type == "reasoning": skip
    return msgs


def _input_to_turn(req_input) -> Message:
    if isinstance(req_input, str):
        return Message(role="user", content=req_input)
    # list of items: take the last user-ish message's text
    text = ""
    for it in req_input:
        if isinstance(it, dict):
            text = (
                _item_text(it) or it.get("content", "")
                if isinstance(it.get("content"), str)
                else _item_text(it)
            )
    return Message(role="user", content=text or "")


async def build_context(
    request: ResponsesRequest, store
) -> Tuple[AgentContext, Optional[str]]:
    """Resolve prior history via the store, assemble the AgentContext the agent runs.
    Returns (ctx, conversation_id). Raises ValueError on previous_response_id/conversation
    conflict (the route maps that to HTTP 400)."""
    history_items: List[Item] = []
    conversation_id = request.conversation
    if request.previous_response_id or request.conversation:
        history_items = await store.resolve_history(
            previous_response_id=request.previous_response_id,
            conversation=request.conversation,
        )
        if request.previous_response_id:
            resp = await store.get_response(request.previous_response_id)
            if resp is not None and resp.conversation_id:
                conversation_id = resp.conversation_id

    ctx = AgentContext(
        system_prompt=request.instructions or DEFAULT_SYSTEM_PROMPT,
        history=items_to_messages(history_items),
        current_turn=_input_to_turn(request.input),
        attachments=[],
        hints=[],
        tools=ToolBox([]),
        run_vars=RunVars(),
    )
    return ctx, conversation_id
```

- [ ] **Step 5: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_builder.py -v` → 5 passed.

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas.py backend/app/builder.py tests/app/test_builder.py
git commit -m "feat(app): ResponsesRequest schema + build_context (history resolution)"
```

---

## Task 3: Responses serializer — SYNC

`AgentEvent` stream → a single OpenAI `Response` object **plus** a parallel list of store-ready item dicts the route persists. READ `backend/agent/core/events.py` and the verified `openai.types.responses` shapes in "Key existing contracts" above.

**Files:** Create `backend/api/protocol/responses_serializer.py`. Test: `tests/app/test_responses_serializer_sync.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/app/test_responses_serializer_sync.py
import sys, os, asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.core.events import (
    RunStarted,
    TextDelta,
    ReasoningDelta,
    ToolStarted,
    ToolCompleted,
    ToolResult,
    RunCompleted,
    RunFailed,
    Usage,
)
from api.protocol.responses_serializer import serialize_response_sync
from openai.types.responses import Response


async def _events(seq):
    for e in seq:
        yield e


def test_sync_text_response_parses_and_persists_items():
    async def run():
        events = _events(
            [
                RunStarted(response_id="resp_1", conversation_id="conv_1"),
                TextDelta(text="Hell"),
                TextDelta(text="o"),
                RunCompleted(
                    usage=Usage(input=3, output=5, total=8),
                    finish_reason="stop",
                ),
            ]
        )
        resp, items = await serialize_response_sync(
            events, model="m", response_id="resp_1", conversation_id="conv_1"
        )
        # Round-trips through the real OpenAI SDK type:
        parsed = Response.model_validate(resp)
        assert parsed.status == "completed"
        assert parsed.output[0].type == "message"
        assert parsed.output[0].content[0].text == "Hello"
        assert parsed.usage.total_tokens == 8
        # Store-ready items: assistant message persisted
        assert any(
            it["type"] == "message"
            and it["role"] == "assistant"
            and it["content"]["text"] == "Hello"
            for it in items
        )

    asyncio.run(run())


def test_sync_tool_call_response_emits_function_call_items():
    async def run():
        events = _events(
            [
                RunStarted(response_id="resp_2"),
                ToolStarted(call_id="c1", name="get"),
                ToolCompleted(call_id="c1", name="get", arguments='{"x":1}'),
                ToolResult(call_id="c1", name="get", ok=True, output="42"),
                TextDelta(text="done"),
                RunCompleted(usage=Usage(input=1, output=1, total=2)),
            ]
        )
        resp, items = await serialize_response_sync(
            events, model="m", response_id="resp_2", conversation_id=None
        )
        parsed = Response.model_validate(resp)
        types = [o.type for o in parsed.output]
        assert "function_call" in types and "message" in types
        fc = next(o for o in parsed.output if o.type == "function_call")
        assert (
            fc.name == "get"
            and fc.arguments == '{"x":1}'
            and fc.call_id == "c1"
        )
        # function_call + function_call_output persisted for history fidelity
        assert any(it["type"] == "function_call" for it in items)
        assert any(
            it["type"] == "function_call_output"
            and it["content"]["output"] == "42"
            for it in items
        )

    asyncio.run(run())


def test_sync_failure_sets_failed_status_and_error():
    async def run():
        events = _events(
            [
                RunStarted(response_id="resp_3"),
                RunFailed(message="kaboom", error_type="llm"),
            ]
        )
        resp, items = await serialize_response_sync(
            events, model="m", response_id="resp_3", conversation_id=None
        )
        parsed = Response.model_validate(resp)
        assert parsed.status == "failed"
        assert parsed.error is not None and "kaboom" in parsed.error.message

    asyncio.run(run())
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: api.protocol.responses_serializer`.

- [ ] **Step 3: Implement `backend/api/protocol/responses_serializer.py`** (sync path + shared item assembly)

```python
from __future__ import annotations
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from agent.core.events import (
    RunStarted,
    TextDelta,
    ReasoningDelta,
    ToolStarted,
    ToolCompleted,
    ToolResult,
    RunCompleted,
    RunFailed,
)
from openai.types.responses import (
    Response,
    ResponseOutputMessage,
    ResponseFunctionToolCall,
    ResponseReasoningItem,
    ResponseUsage,
)
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import Summary, Content
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from openai.types.responses.response_error import ResponseError


def _usage(u) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=u.input,
        output_tokens=u.output,
        total_tokens=u.total,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


class _Assembler:
    """Consumes an AgentEvent stream into OpenAI Response output items + store items.
    Shared by the sync and streaming serializers."""

    def __init__(
        self, model: str, response_id: str, conversation_id: Optional[str]
    ):
        self.model = model
        self.response_id = response_id
        self.conversation_id = conversation_id
        self.output: List[Any] = []  # OpenAI output items
        self.store_items: List[
            Dict
        ] = []  # store-ready dicts {type, role, content}
        self.text = ""
        self.reasoning = ""
        self.usage = None
        self.status = "completed"
        self.error: Optional[ResponseError] = None

    def _msg_item_id(self) -> str:
        return f"msg_{self.response_id}"

    def on_text(self, text: str):
        self.text += text

    def on_reasoning(self, text: str):
        self.reasoning += text

    def on_tool_completed(self, call_id: str, name: str, arguments: str):
        self.output.append(
            ResponseFunctionToolCall(
                id=f"fc_{call_id}",
                call_id=call_id,
                name=name,
                arguments=arguments or "",
                type="function_call",
                status="completed",
            )
        )
        self.store_items.append(
            {
                "type": "function_call",
                "role": None,
                "content": {
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments or "",
                },
            }
        )

    def on_tool_result(
        self, call_id: str, output: Optional[str], error: Optional[str]
    ):
        self.store_items.append(
            {
                "type": "function_call_output",
                "role": None,
                "content": {
                    "call_id": call_id,
                    "output": output if output is not None else (error or ""),
                },
            }
        )

    def on_failed(self, message: str):
        self.status = "failed"
        self.error = ResponseError(code="server_error", message=message)

    def finalize(self, usage) -> None:
        # reasoning item (if any) first, then the assistant message
        if self.reasoning:
            self.output.insert(
                0,
                ResponseReasoningItem(
                    id=f"rs_{self.response_id}",
                    type="reasoning",
                    status="completed",
                    summary=[],
                    content=[
                        Content(text=self.reasoning, type="reasoning_text")
                    ],
                ),
            )
            self.store_items.append(
                {
                    "type": "reasoning",
                    "role": None,
                    "content": {"text": self.reasoning},
                }
            )
        if self.text or self.status == "completed":
            self.output.append(
                ResponseOutputMessage(
                    id=self._msg_item_id(),
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        ResponseOutputText(
                            annotations=[], text=self.text, type="output_text"
                        )
                    ],
                )
            )
            self.store_items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": {"text": self.text},
                }
            )
        if usage is not None:
            self.usage = _usage(usage)

    def to_response(self) -> Response:
        return Response(
            id=self.response_id,
            created_at=time.time(),
            model=self.model,
            object="response",
            output=self.output,
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            status=self.status,
            usage=self.usage,
            error=self.error,
            previous_response_id=None,
            conversation=(
                {"id": self.conversation_id} if self.conversation_id else None
            ),
        )


async def serialize_response_sync(
    events: AsyncIterator,
    *,
    model: str,
    response_id: str,
    conversation_id: Optional[str],
) -> Tuple[dict, List[Dict]]:
    """Consume the full AgentEvent stream and return (response_dict, store_items).
    response_dict is `Response.model_dump()` (JSON-ready, OpenAI-conformant).
    """
    asm = _Assembler(model, response_id, conversation_id)
    usage = None
    async for ev in events:
        if isinstance(ev, TextDelta):
            asm.on_text(ev.text)
        elif isinstance(ev, ReasoningDelta):
            asm.on_reasoning(ev.text)
        elif isinstance(ev, ToolCompleted):
            asm.on_tool_completed(ev.call_id, ev.name, ev.arguments)
        elif isinstance(ev, ToolResult):
            asm.on_tool_result(ev.call_id, ev.output, ev.error)
        elif isinstance(ev, RunCompleted):
            usage = ev.usage
        elif isinstance(ev, RunFailed):
            asm.on_failed(ev.message)
        # RunStarted, ToolStarted: no sync effect
    asm.finalize(usage)
    return asm.to_response().model_dump(mode="json"), asm.store_items
```

NOTE: `ResponseError.code` is a constrained literal — always `"server_error"`. The store-item content shapes here (`{"text": ...}`, `{"call_id","name","arguments"}`, `{"call_id","output"}`) MUST match what `app/builder.py:items_to_messages` reads (Task 2). If a field name diverges, fix it in BOTH places.

- [ ] **Step 4: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_responses_serializer_sync.py -v` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/api/protocol/responses_serializer.py tests/app/test_responses_serializer_sync.py
git commit -m "feat(protocol): AgentEvent -> OpenAI Responses sync serializer"
```

---

## Task 4: Responses serializer — STREAM

Add `serialize_response_stream` to the same module: yield SSE `response.*` event JSON strings in OpenAI order, accumulating the final `Response` + store items into a caller-provided `sink` dict (mirrors the agent's sink pattern, so the route persists after the stream ends).

**Files:** Modify `backend/api/protocol/responses_serializer.py`. Test: `tests/app/test_responses_serializer_stream.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/app/test_responses_serializer_stream.py
import sys, os, asyncio, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.core.events import (
    RunStarted,
    TextDelta,
    ToolStarted,
    ToolCompleted,
    ToolResult,
    RunCompleted,
    RunFailed,
    Usage,
)
from api.protocol.responses_serializer import serialize_response_stream
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter

_ADAPTER = TypeAdapter(ResponseStreamEvent)


def _parse(lines):
    evs = []
    for ln in lines:
        for part in ln.splitlines():
            if part.startswith("data:"):
                payload = part[len("data:") :].strip()
                if payload and payload != "[DONE]":
                    evs.append(_ADAPTER.validate_python(json.loads(payload)))
    return evs


async def _events(seq):
    for e in seq:
        yield e


def test_stream_text_event_order_and_sink():
    async def run():
        sink = {}
        gen = serialize_response_stream(
            _events(
                [
                    RunStarted(response_id="resp_1", conversation_id="conv_1"),
                    TextDelta(text="Hi"),
                    TextDelta(text="!"),
                    RunCompleted(usage=Usage(input=1, output=1, total=2)),
                ]
            ),
            model="m",
            response_id="resp_1",
            conversation_id="conv_1",
            sink=sink,
        )
        lines = [chunk async for chunk in gen]
        evs = _parse(lines)
        types = [e.type for e in evs]
        assert types[0] == "response.created"
        assert "response.in_progress" in types
        assert "response.output_text.delta" in types
        assert types[-1] == "response.completed"
        # full text reconstructable from deltas
        text = "".join(
            e.delta for e in evs if e.type == "response.output_text.delta"
        )
        assert text == "Hi!"
        # sink carries the final response + persisted items
        assert sink["response"]["status"] == "completed"
        assert any(it["type"] == "message" for it in sink["items"])

    asyncio.run(run())


def test_stream_tool_call_events():
    async def run():
        sink = {}
        gen = serialize_response_stream(
            _events(
                [
                    RunStarted(response_id="resp_2"),
                    ToolStarted(call_id="c1", name="get"),
                    ToolCompleted(
                        call_id="c1", name="get", arguments='{"x":1}'
                    ),
                    ToolResult(call_id="c1", name="get", ok=True, output="42"),
                    TextDelta(text="ok"),
                    RunCompleted(usage=Usage(input=1, output=1, total=2)),
                ]
            ),
            model="m",
            response_id="resp_2",
            conversation_id=None,
            sink=sink,
        )
        evs = _parse([c async for c in gen])
        types = [e.type for e in evs]
        assert "response.function_call_arguments.done" in types
        done = next(
            e for e in evs if e.type == "response.function_call_arguments.done"
        )
        assert done.arguments == '{"x":1}' and done.name == "get"

    asyncio.run(run())


def test_stream_failure_emits_failed_event():
    async def run():
        sink = {}
        gen = serialize_response_stream(
            _events(
                [
                    RunStarted(response_id="resp_3"),
                    RunFailed(message="boom", error_type="llm"),
                ]
            ),
            model="m",
            response_id="resp_3",
            conversation_id=None,
            sink=sink,
        )
        evs = _parse([c async for c in gen])
        assert evs[-1].type == "response.failed"
        assert sink["response"]["status"] == "failed"

    asyncio.run(run())
```

- [ ] **Step 2: Run to verify failure** — `ImportError: cannot import name 'serialize_response_stream'`.

- [ ] **Step 3: Add `serialize_response_stream` to `backend/api/protocol/responses_serializer.py`**

Append these imports to the existing import block:
```python
from openai.types.responses import (
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
)
```

Append the streaming serializer:
```python
def _sse(event) -> str:
    return f"data: {event.model_dump_json()}\n\n"


async def serialize_response_stream(
    events: AsyncIterator,
    *,
    model: str,
    response_id: str,
    conversation_id: Optional[str],
    sink: Dict,
) -> AsyncIterator[str]:
    """AgentEvent stream -> SSE `response.*` event strings (OpenAI order). On completion,
    sink["response"] = final Response.model_dump(mode="json") and sink["items"] = store items.
    """
    asm = _Assembler(model, response_id, conversation_id)
    seq = 0

    def nxt() -> int:
        nonlocal seq
        seq += 1
        return seq

    # response.created + response.in_progress
    yield _sse(
        ResponseCreatedEvent(
            response=asm.to_response(),
            sequence_number=nxt(),
            type="response.created",
        )
    )
    yield _sse(
        ResponseInProgressEvent(
            response=asm.to_response(),
            sequence_number=nxt(),
            type="response.in_progress",
        )
    )

    output_index = 0
    msg_open = False
    msg_item_id = asm._msg_item_id()
    usage = None

    async for ev in events:
        if isinstance(ev, TextDelta):
            if not msg_open:
                # open a message output item + a text content part
                from openai.types.responses import ResponseOutputMessage as _M
                from openai.types.responses.response_output_text import (
                    ResponseOutputText as _T,
                )

                placeholder = _M(
                    id=msg_item_id,
                    role="assistant",
                    status="in_progress",
                    type="message",
                    content=[],
                )
                yield _sse(
                    ResponseOutputItemAddedEvent(
                        item=placeholder,
                        output_index=output_index,
                        sequence_number=nxt(),
                        type="response.output_item.added",
                    )
                )
                yield _sse(
                    ResponseContentPartAddedEvent(
                        content_index=0,
                        item_id=msg_item_id,
                        output_index=output_index,
                        part=_T(annotations=[], text="", type="output_text"),
                        sequence_number=nxt(),
                        type="response.content_part.added",
                    )
                )
                msg_open = True
            asm.on_text(ev.text)
            yield _sse(
                ResponseTextDeltaEvent(
                    content_index=0,
                    delta=ev.text,
                    item_id=msg_item_id,
                    logprobs=[],
                    output_index=output_index,
                    sequence_number=nxt(),
                    type="response.output_text.delta",
                )
            )
        elif isinstance(ev, ReasoningDelta):
            asm.on_reasoning(ev.text)
            # surface reasoning text as a streaming delta (item assembled at finalize)
            from openai.types.responses import (
                ResponseReasoningTextDeltaEvent as _RD,
            )

            yield _sse(
                _RD(
                    content_index=0,
                    delta=ev.text,
                    item_id=f"rs_{response_id}",
                    output_index=output_index,
                    sequence_number=nxt(),
                    type="response.reasoning_text.delta",
                )
            )
        elif isinstance(ev, ToolStarted):
            asm_fc_index = output_index if not msg_open else output_index + 1
            from openai.types.responses import ResponseFunctionToolCall as _FC

            yield _sse(
                ResponseOutputItemAddedEvent(
                    item=_FC(
                        id=f"fc_{ev.call_id}",
                        call_id=ev.call_id,
                        name=ev.name,
                        arguments="",
                        type="function_call",
                        status="in_progress",
                    ),
                    output_index=asm_fc_index,
                    sequence_number=nxt(),
                    type="response.output_item.added",
                )
            )
        elif isinstance(ev, ToolCompleted):
            asm.on_tool_completed(ev.call_id, ev.name, ev.arguments)
            fc_index = output_index if not msg_open else output_index + 1
            yield _sse(
                ResponseFunctionCallArgumentsDeltaEvent(
                    delta=ev.arguments or "",
                    item_id=f"fc_{ev.call_id}",
                    output_index=fc_index,
                    sequence_number=nxt(),
                    type="response.function_call_arguments.delta",
                )
            )
            yield _sse(
                ResponseFunctionCallArgumentsDoneEvent(
                    arguments=ev.arguments or "",
                    item_id=f"fc_{ev.call_id}",
                    name=ev.name,
                    output_index=fc_index,
                    sequence_number=nxt(),
                    type="response.function_call_arguments.done",
                )
            )
            from openai.types.responses import ResponseFunctionToolCall as _FC

            yield _sse(
                ResponseOutputItemDoneEvent(
                    item=_FC(
                        id=f"fc_{ev.call_id}",
                        call_id=ev.call_id,
                        name=ev.name,
                        arguments=ev.arguments or "",
                        type="function_call",
                        status="completed",
                    ),
                    output_index=fc_index,
                    sequence_number=nxt(),
                    type="response.output_item.done",
                )
            )
        elif isinstance(ev, ToolResult):
            asm.on_tool_result(ev.call_id, ev.output, ev.error)
        elif isinstance(ev, RunCompleted):
            usage = ev.usage
        elif isinstance(ev, RunFailed):
            asm.on_failed(ev.message)

    # close an open message item
    if msg_open:
        from openai.types.responses.response_output_text import (
            ResponseOutputText as _T,
        )

        yield _sse(
            ResponseTextDoneEvent(
                content_index=0,
                item_id=msg_item_id,
                logprobs=[],
                output_index=output_index,
                sequence_number=nxt(),
                text=asm.text,
                type="response.output_text.done",
            )
        )
        yield _sse(
            ResponseContentPartDoneEvent(
                content_index=0,
                item_id=msg_item_id,
                output_index=output_index,
                part=_T(annotations=[], text=asm.text, type="output_text"),
                sequence_number=nxt(),
                type="response.content_part.done",
            )
        )

    asm.finalize(usage)
    if msg_open:
        from openai.types.responses import ResponseOutputMessage as _M
        from openai.types.responses.response_output_text import (
            ResponseOutputText as _T,
        )

        yield _sse(
            ResponseOutputItemDoneEvent(
                item=_M(
                    id=msg_item_id,
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[
                        _T(annotations=[], text=asm.text, type="output_text")
                    ],
                ),
                output_index=output_index,
                sequence_number=nxt(),
                type="response.output_item.done",
            )
        )

    final = asm.to_response()
    if asm.status == "failed":
        yield _sse(
            ResponseFailedEvent(
                response=final, sequence_number=nxt(), type="response.failed"
            )
        )
    else:
        yield _sse(
            ResponseCompletedEvent(
                response=final,
                sequence_number=nxt(),
                type="response.completed",
            )
        )

    sink["response"] = final.model_dump(mode="json")
    sink["items"] = asm.store_items
```

NOTE on ordering/indices: this is the trickiest code in the plan. The conformance gate is the test parsing every event via `TypeAdapter(ResponseStreamEvent)` and reconstructing the text. If the SDK rejects an event (missing/incorrect field) or an index assertion fails, READ the SDK event class and adjust field values — keep the OpenAI event SEQUENCE (created → in_progress → [item.added → content_part.added → text.delta* → text.done → content_part.done → item.done] and function_call item.added → args.delta → args.done → item.done → completed/failed). Do not invent fields. The `output_index` bookkeeping for interleaved message + function_call items only needs to keep distinct items on distinct indices and parse cleanly; perfect index fidelity to OpenAI is not asserted, SDK-parseability + ordering + text reconstruction is.

- [ ] **Step 4: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_responses_serializer_stream.py -v` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/api/protocol/responses_serializer.py tests/app/test_responses_serializer_stream.py
git commit -m "feat(protocol): AgentEvent -> OpenAI Responses streaming serializer (SSE)"
```

---

## Task 5: Routes — `/v1/responses` (+ `/v1/chat/completions` shim)

Wire the endpoints. POST resolves input via the store, builds context, runs the agent, serializes (stream or sync), and persists iff `store=true`. GET/DELETE are store-backed. A small `deps.py` exposes app state (store, agent factory) so tests can inject a fake LLM. READ `agent/agent.py` `Agent.__init__` for the exact constructor, and `api/protocol/chat_serializer.py` for the chat functions.

**Files:** Create `backend/app/deps.py`, `backend/app/routes/__init__.py`, `backend/app/routes/responses.py`, `backend/app/routes/chat.py`. Test: `tests/app/test_routes_responses.py`.

- [ ] **Step 1: Write the failing test** (FastAPI app assembled in-test with InMemoryStore + a fake echo LLM)

```python
# tests/app/test_routes_responses.py
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.store.memory import InMemoryStore
from app.routes.responses import router as responses_router
from app.deps import AppState
from common.llm.models import TextChunk


class _EchoLLM:
    """Yields the user's last message text back as a single assistant chunk + usage."""

    async def astream(self, messages, tools=None, **kwargs):
        last = ""
        for m in messages:
            if m.get("role") == "user":
                last = m.get("content") or ""

        class _U:
            prompt_tokens, completion_tokens, total_tokens = 1, 1, 2

        yield TextChunk(delta=f"echo:{last}", usage=None)
        yield TextChunk(delta="", usage=_U())


def _client():
    app = FastAPI()
    app.state.app_state = AppState(
        store=InMemoryStore(), llm=_EchoLLM(), default_model="m"
    )
    app.include_router(responses_router)
    return TestClient(app)


def test_post_sync_creates_and_persists_response():
    c = _client()
    r = c.post("/v1/responses", json={"input": "hello", "stream": False})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "response" and body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"] == "echo:hello"
    rid = body["id"]
    got = c.get(f"/v1/responses/{rid}")
    assert got.status_code == 200 and got.json()["id"] == rid


def test_post_stream_returns_sse_events():
    c = _client()
    with c.stream(
        "POST", "/v1/responses", json={"input": "hi", "stream": True}
    ) as r:
        assert r.status_code == 200
        raw = "".join(chunk for chunk in r.iter_text())
    assert "response.created" in raw and "response.completed" in raw
    assert "response.output_text.delta" in raw


def test_previous_response_id_links_history():
    c = _client()
    first = c.post(
        "/v1/responses", json={"input": "one", "stream": False}
    ).json()
    second = c.post(
        "/v1/responses",
        json={
            "input": "two",
            "stream": False,
            "previous_response_id": first["id"],
        },
    ).json()
    # second turn shares the conversation of the first
    assert second["conversation"]["id"] == first["conversation"]["id"]


def test_store_false_is_not_retrievable():
    c = _client()
    body = c.post(
        "/v1/responses",
        json={"input": "ephemeral", "stream": False, "store": False},
    ).json()
    assert c.get(f"/v1/responses/{body['id']}").status_code == 404


def test_delete_response():
    c = _client()
    body = c.post("/v1/responses", json={"input": "x", "stream": False}).json()
    assert c.delete(f"/v1/responses/{body['id']}").status_code == 200
    assert c.get(f"/v1/responses/{body['id']}").status_code == 404


def test_conflicting_ids_returns_400():
    c = _client()
    a = c.post("/v1/responses", json={"input": "a", "stream": False}).json()
    # fabricate a conflicting conversation id
    r = c.post(
        "/v1/responses",
        json={
            "input": "b",
            "stream": False,
            "previous_response_id": a["id"],
            "conversation": "conv_other",
        },
    )
    assert r.status_code == 400
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: app.deps`.

- [ ] **Step 3: Implement `backend/app/deps.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from fastapi import Request
from agent.agent import Agent
from agent.budgeting import AgentMessageManager


@dataclass
class AppState:
    store: object  # ResponseStore
    llm: object  # has .astream(messages, tools)
    default_model: str
    context_window: int = 110000
    max_output_tokens: int = 8000

    def make_agent(self) -> Agent:
        # Verified signature: Agent(llm, max_steps=..., budget: Optional[AgentMessageManager]=None).
        # Pass an explicit budget from AppState's window so fake/echo test LLMs (which
        # lack a `context_window` attr) don't get a zero-width budget that truncates input.
        return Agent(
            llm=self.llm,
            budget=AgentMessageManager(
                context_window=self.context_window,
                max_output_tokens=self.max_output_tokens,
            ),
        )


def get_state(request: Request) -> AppState:
    return request.app.state.app_state
```

NOTE: `Agent.__init__(self, llm, max_steps=MAX_RECURSION_STEPS, budget=None)` is verified — it builds its own budget from `getattr(llm, "context_window", 0)`/`getattr(llm, "max_tokens", 0)` if none is passed. We pass an explicit `budget` so it never degenerates to a zero-width window. `Agent.run(ctx)` is `async` and returns the event generator; `await agent.run(ctx)` then `async for` over it (as the routes do).

- [ ] **Step 4: Implement `backend/app/routes/responses.py`**

```python
from __future__ import annotations
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from app.schemas import ResponsesRequest
from app.builder import build_context
from app.deps import AppState, get_state
from app.store.base import Item, StoredResponse
from api.protocol.responses_serializer import (
    serialize_response_sync,
    serialize_response_stream,
)

router = APIRouter()


def _rid() -> str:
    return f"resp_{uuid.uuid4().hex}"


def _user_input_items(request: ResponsesRequest, response_id: str) -> list:
    """The user's turn, stored as a message item (history source of truth)."""
    if isinstance(request.input, str):
        text = request.input
    else:
        text = ""
        for it in request.input:
            if isinstance(it, dict) and isinstance(it.get("content"), str):
                text = it["content"]
    return [
        Item(
            type="message",
            role="user",
            content={"text": text},
            response_id=response_id,
        )
    ]


async def _ensure_conversation(state: AppState, conversation_id):
    if conversation_id:
        return conversation_id
    conv = await state.store.create_conversation()
    return conv.id


async def _persist(
    state: AppState,
    request: ResponsesRequest,
    response_id: str,
    conversation_id: str,
    store_items: list,
    status: str,
    usage: dict | None,
):
    items = _user_input_items(request, response_id)
    for d in store_items:
        items.append(
            Item(
                type=d["type"],
                role=d.get("role"),
                content=d["content"],
                response_id=response_id,
            )
        )
    await state.store.append_items(conversation_id, items)
    await state.store.save_response(
        StoredResponse(
            id=response_id,
            conversation_id=conversation_id,
            model=request.model or state.default_model,
            status=status,
            usage=usage,
            previous_response_id=request.previous_response_id,
        )
    )


@router.post("/v1/responses")
async def create_response(
    request: ResponsesRequest,
    req: Request,
    state: AppState = Depends(get_state),
):
    if not request.model:
        request.model = state.default_model
    try:
        ctx, conversation_id = await build_context(request, state.store)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    response_id = _rid()
    agent = state.make_agent()
    events = await agent.run(ctx)

    if request.stream:

        async def gen():
            sink = {}
            conv_id = (
                await _ensure_conversation(state, conversation_id)
                if request.store
                else conversation_id
            )
            async for chunk in serialize_response_stream(
                events,
                model=request.model,
                response_id=response_id,
                conversation_id=conv_id,
                sink=sink,
            ):
                yield chunk
            if request.store and sink.get("response"):
                r = sink["response"]
                await _persist(
                    state,
                    request,
                    response_id,
                    conv_id,
                    sink["items"],
                    r["status"],
                    r.get("usage"),
                )

        return StreamingResponse(gen(), media_type="text/event-stream")

    conv_id = (
        await _ensure_conversation(state, conversation_id)
        if request.store
        else conversation_id
    )
    resp_dict, store_items = await serialize_response_sync(
        events,
        model=request.model,
        response_id=response_id,
        conversation_id=conv_id,
    )
    if request.store:
        await _persist(
            state,
            request,
            response_id,
            conv_id,
            store_items,
            resp_dict["status"],
            resp_dict.get("usage"),
        )
    return JSONResponse(resp_dict)


@router.get("/v1/responses/{response_id}")
async def get_response(response_id: str, state: AppState = Depends(get_state)):
    stored = await state.store.get_response(response_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="response not found")
    return JSONResponse(
        {
            "id": stored.id,
            "object": "response",
            "status": stored.status,
            "model": stored.model,
            "conversation": (
                {"id": stored.conversation_id}
                if stored.conversation_id
                else None
            ),
            "usage": stored.usage,
            "error": stored.error,
            "previous_response_id": stored.previous_response_id,
        }
    )


@router.delete("/v1/responses/{response_id}")
async def delete_response(
    response_id: str, state: AppState = Depends(get_state)
):
    stored = await state.store.get_response(response_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="response not found")
    await state.store.delete_response(response_id)
    return JSONResponse(
        {"id": response_id, "object": "response.deleted", "deleted": True}
    )
```

- [ ] **Step 5: Implement `backend/app/routes/chat.py`** (thin `/v1/chat/completions` shim)

```python
from __future__ import annotations
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from agent.context import AgentContext, RunVars
from agent.message import from_thread
from agent.tools.base import ToolBox
from app.deps import AppState, get_state
from api.protocol.chat_serializer import (
    serialize_chat_stream,
    serialize_chat_sync_with_effects,
)
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional

router = APIRouter()


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: Optional[str] = None
    messages: List[Dict[str, Any]] = []
    stream: bool = True
    system: Optional[str] = None


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest, state: AppState = Depends(get_state)
):
    model = request.model or state.default_model
    msgs = from_thread(request.messages)
    current_turn = msgs[-1] if msgs else None
    history = msgs[:-1]
    ctx = AgentContext(
        system_prompt=request.system or "You are a helpful assistant.",
        history=history,
        current_turn=current_turn,
        attachments=[],
        hints=[],
        tools=ToolBox([]),
        run_vars=RunVars(),
    )
    agent = state.make_agent()
    events = await agent.run(ctx)
    if request.stream:

        async def gen():
            async for chunk in serialize_chat_stream(events, model=model):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")
    result = await serialize_chat_sync_with_effects(events, model=model)
    return JSONResponse(result)
```

NOTE: `serialize_chat_stream`/`serialize_chat_sync_with_effects` signatures were verified to take `events, *, model` (plus optional effect kwargs that default to None/off). READ `api/protocol/chat_serializer.py` and pass only what's needed; if `serialize_chat_stream` already yields full `data: ...` lines, don't double-wrap. Adapt the wrapping to the real function's output. Create empty `backend/app/routes/__init__.py`.

- [ ] **Step 6: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_routes_responses.py -v` → 6 passed.

- [ ] **Step 7: Commit**

```bash
git add backend/app/deps.py backend/app/routes/ tests/app/test_routes_responses.py
git commit -m "feat(app): /v1/responses (stream+sync, GET, DELETE) + /v1/chat/completions shim"
```

---

## Task 6: `app/main.py` — boot the lean service

A minimal FastAPI app: lifespan runs `create_all` (when SQL), builds `AppState` (store from `STORE_BACKEND`, `LeanLLM` from settings), mounts routes. A lean-boot test exercises the whole path with `STORE_BACKEND=memory` + a fake LLM and asserts `/v1/responses` works with zero RAG/tokenizer/trace files.

**Files:** Create `backend/app/main.py`. Test: `tests/app/test_main_boot.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/app/test_main_boot.py
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from fastapi.testclient import TestClient


def test_app_boots_in_memory_and_serves(monkeypatch):
    monkeypatch.setenv("STORE_BACKEND", "memory")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("DEFAULT_MODEL", "m")
    import importlib
    import app.main as m

    importlib.reload(m)
    # Inject a fake LLM so we don't hit the network.
    from common.llm.models import TextChunk

    class _EchoLLM:
        async def astream(self, messages, tools=None, **kwargs):
            class _U:
                prompt_tokens = completion_tokens = total_tokens = 1

            yield TextChunk(delta="echo", usage=None)
            yield TextChunk(delta="", usage=_U())

    with TestClient(m.app) as c:
        c.app.state.app_state.llm = _EchoLLM()
        r = c.post("/v1/responses", json={"input": "hi", "stream": False})
        assert r.status_code == 200
        assert r.json()["output"][0]["content"][0]["text"] == "echo"
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: app.main` (or, if `app/main.py` already exists as a legacy heavy app, this test imports the wrong app — see Step 3 note).

- [ ] **Step 3: Implement `backend/app/main.py`**

IMPORTANT: `backend/app/main.py` may ALREADY EXIST as the legacy heavy app (it imports RAG/llamaindex/trace). READ it first. If it exists and is the legacy app, DO NOT overwrite it — instead create the lean app as `backend/app/lean_main.py` and update the test import to `app.lean_main`. If it does not exist (or is trivial), create `app/main.py`. Decide based on what you read, and keep the test's import path in sync with the file you create.

```python
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import get_settings
from app.deps import AppState
from app.db import make_engine, create_all
from app.store.memory import InMemoryStore
from app.store.sql import SqlStore
from app.llm import LeanLLM
from app.routes.responses import router as responses_router
from app.routes.chat import router as chat_router


def _build_state(settings) -> AppState:
    if settings.store_backend == "memory":
        store = InMemoryStore()
    else:
        engine = make_engine(settings.db_url)
        store = SqlStore(engine)
    llm = LeanLLM(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.default_model,
    )
    return AppState(store=store, llm=llm, default_model=settings.default_model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.store_backend != "memory":
        # ensure the data dir exists for file-based sqlite
        if (
            settings.db_url.startswith("sqlite")
            and ":memory:" not in settings.db_url
        ):
            os.makedirs("./data", exist_ok=True)
        engine = make_engine(settings.db_url)
        await create_all(engine)
    app.state.app_state = _build_state(settings)
    yield


app = FastAPI(title="Lean Agent Service", lifespan=lifespan)
app.include_router(responses_router)
app.include_router(chat_router)
```

NOTE: for `store_backend != "memory"`, `_build_state` creates a second engine; that's fine for SQLite-file/Postgres (shared DB), but NOT for `:memory:` (each engine = separate DB). The lean-boot test uses `memory` store so this isn't exercised; if you later test SQL-backed boot, build ONE engine in `lifespan` and pass it into `AppState`/`SqlStore`. Keep it simple now.

- [ ] **Step 4: Run to verify pass** — `cd backend && python -m pytest ../tests/app/test_main_boot.py -v` → 1 passed.

- [ ] **Step 5: Full foundation + service check + commit**

Run: `cd backend && python -m pytest ../tests/app/ ../tests/agent/ -q` → all pass.
```bash
git add backend/app/main.py tests/app/test_main_boot.py
git commit -m "feat(app): lean FastAPI main (create_all lifespan, store/LLM wiring, routes)"
```

---

## Notes for the implementer

- **Reuse, don't reinvent:** `update_tool_calls` (tool-call coalescing), `from_thread` (message normalization), and the whole `chat_serializer` are existing and verified — import them. Only `LeanLLM` and the responses serializer are genuinely new logic.
- **Store-item content shapes are a contract** between `responses_serializer` (writer) and `builder.items_to_messages` (reader): `message`→`{"text","role on Item"}`, `function_call`→`{"call_id","name","arguments"}`, `function_call_output`→`{"call_id","output"}`, `reasoning`→`{"text"}`. Keep them identical in both files.
- **OpenAI conformance is the gate, not guesswork:** every serializer test parses output through the real `openai.types.responses` types (`Response.model_validate` / `TypeAdapter(ResponseStreamEvent)`). If the SDK rejects a constructed object, READ that class's fields and fix — never loosen the test.
- **Agent constructor:** Task 5/6 wire `Agent` + `AgentMessageManager`. The exact `Agent.__init__` signature is the one real unknown — READ `agent/agent.py` and adapt `AppState.make_agent`. Everything else is pinned by verified contracts above.
- **No RAG/tools yet:** `ToolBox([])` everywhere. The frontend's `enable_agent`/`kb_ids`/`mcp_ids` are accepted-and-ignored (`extra="ignore"`). Tools return as plugins in a later plan.
- **Do NOT touch the legacy backend** beyond reading it. If `app/main.py` is the legacy app, use `app/lean_main.py` (Task 6, Step 3). The lean service must boot with zero RAG/llamaindex/tokenizer/trace files present.
- **Frontend migration to `/v1/responses` is a separate plan** (next), written after this service is green.
