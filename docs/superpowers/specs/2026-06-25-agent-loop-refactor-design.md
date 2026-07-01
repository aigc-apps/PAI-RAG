# Agent Loop Refactor — Design

**Date:** 2026-06-25
**Status:** Approved (pending spec review)
**Scope:** `backend/agent/` and its two callers (`backend/api/v1/chat.py`, `backend/service/agent/agent_service.py`)

## Problem

The current agent layer is hard to reason about and hard to debug. The recent
"uploaded file never reaches the model" bug took many iterations to locate
because the answer to a single question — *what does the model actually
receive?* — is assembled across four files:

- `api/v1/chat.py` captures the user message and builds `AgentState`.
- `service/agent/agent_service.py::parse_attachment_tools` **mutates** the user
  message in place (appends attachment content / tool hints).
- `agent/state.py::convert_thread_messages` flattens content arrays and
  normalizes tool-call/tool-result shapes.
- `agent/react_agent.py::run_async` prepends a `[System Time:…]` prefix and then
  fits the result to the token budget.

On top of that, `run_async` is a single ~240-line async generator that
interleaves message prep, LLM streaming, an "intent-nudge" buffering workaround,
tool dispatch, `return_direct` handling, and token fitting. It is nearly
impossible to unit-test and difficult to read.

## Goals

- A clean-slate, minimal `Agent` abstraction built around three concepts the
  team already reaches for: **loop**, **message**, **context**.
- **One** place that constructs model input, pure and logged, so "what does the
  model see?" is answered by inspecting a single object and a single function.
- A thin, testable loop (call → dispatch → append → stop on plain text).
- Preserve load-bearing behavior; shed workarounds.

## Non-Goals

- No rewrite of the ~12 tool factories under `tools/` (we keep adapting
  llama_index `FunctionTool`).
- No change to the SSE wire format or the serializer in `common/llm/utils.py`.
- No change to RAG / attachment *extraction* (only where its output is assembled
  into model input).

## Decisions (from brainstorming)

1. **Clean-slate** minimal Agent (not a behavior-preserving restructure).
2. **Keep:** token budgeting + summarization, `return_direct` tools, guardrail
   input/output checks, streaming-to-UI, OpenTelemetry tracing.
   **Drop:** intent-nudge buffering — **entirely** (no minimal nudge retained).
3. **Single seam:** `agent_service` populates an `AgentContext`; the `Agent` has
   one pure `build_messages(context)` that is the only place model input is
   constructed. The loop never mutates messages.
4. **Approach A:** three new core types + thin loop; reuse existing
   `FunctionTool` adapters and streaming chunk types; slim `message_manager`
   into a `budget.fit()` call.

Reference implementations that informed this (flat message list, thin loop,
context object, budgeting/guardrails as separate steps): claude-code
(single-threaded master loop, flat history, rolling compaction), nanobot
(orchestration <500 lines, "memory/skills as context"), pydantic-ai (typed
`RunContext` carrying deps + state + messages + tracer).

## Architecture

### Module layout (`backend/agent/`)

| File | Responsibility | Replaces |
|---|---|---|
| `message.py` | `Message` type + `from_thread()` (wire→Message) + `keep_last_rounds()` | `state.py` (`convert_thread_messages`, `AgentState`) |
| `context.py` | `AgentContext` + `RunVars` | today's thin `RunContext` |
| `agent.py` | `Agent`: `build_messages()` + `run()` loop | `react_agent.py` |
| `tools.py` | `ToolBox`: name→tool map, OpenAI schema, `dispatch()`, `is_return_direct()` | `tool_utils.py` + dispatch helpers in `react_agent.py` |
| `budgeting.py` | `AgentMessageManager.fit(messages)` — kept, slimmed | `message_manager.py` (renamed) |
| `events.py` | re-exports existing chunk types as `Event` | — |
| `prompts.py` | unchanged | — |

`react_agent.py` and `state.py` are deleted at the end of migration.

### Core types

```text
# message.py
@dataclass
class Message:
    role: str                                      # system | user | assistant | tool
    content: str | list[ContentPart] | None = None # str for text; list only for vision/multimodal
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    def to_wire(self) -> dict: ...
    @classmethod
    def from_wire(cls, d: dict) -> "Message": ...

def from_thread(raw: list[dict]) -> list[Message]   # normalize messy incoming shapes, once
def keep_last_rounds(msgs: list[Message], n: int) -> list[Message]

# context.py — the single inspectable object
@dataclass
class AgentContext:
    system_prompt: str
    history: list[Message]          # prior turns, already trimmed
    current_turn: Message           # the live user message — NEVER mutated
    attachments: list[Attachment]   # inline file text / images, resolved by agent_service as DATA
    tools: ToolBox
    run_vars: RunVars               # current_datetime, etc.

# agent.py
class Agent:
    def __init__(self, llm, *, max_steps, budget, ...): ...
    def build_messages(self, ctx: AgentContext) -> list[Message]: ...   # ONLY assembly point; logs output
    async def run(self, ctx: AgentContext) -> AsyncIterator[Event]: ... # thin loop
```

`Attachment` is a small tagged record, e.g. `{kind: "file_text" | "image", name, body}`.

### The single seam

```text
def build_messages(self, ctx) -> list[Message]:
    msgs = [Message("system", ctx.system_prompt)]
    msgs += ctx.history
    msgs.append(render_current_turn(ctx.current_turn, ctx.attachments, ctx.run_vars))
    logger.info("[agent] model input: %s", summarize(msgs))   # one line, full picture
    return msgs
```

`render_current_turn` is the one place where the `[System Time:…]` header, the
user's text, and `<attached_file name="…">…</attached_file>` blocks come
together. It replaces both the time-prefix logic in `react_agent` and the
in-place message mutation in `parse_attachment_tools`.

### The loop

```text
async def run(self, ctx) -> AsyncIterator[Event]:
    with trace_span("agent.run"):                       # tracing: kept
        messages = self.build_messages(ctx)
        for step in range(self.max_steps):
            messages = self.budget.fit(messages)         # budgeting: kept (one call)
            assistant, tool_calls = await self._stream_turn(messages, ctx.tools)
            #  ^ yields TextDelta / Reasoning events directly — NO buffering
            messages.append(assistant)
            if not tool_calls:
                return                                   # plain text → done
            for tc in tool_calls:
                result = await ctx.tools.dispatch(tc)    # parse args + retry + capture
                messages.append(result.message)
                yield ToolResultEvent(result)
                if ctx.tools.is_return_direct(tc.name) and result.ok:
                    yield result.as_direct(); return     # return_direct: kept
        yield max_steps_notice()
```

### Where kept features live

- **Token budgeting** → `self.budget.fit(messages)`, one call per step. The loop
  is ignorant of its internals (grouping/summarization/truncation).
- **return_direct** → a `ToolBox` flag check after dispatch. No scattering.
- **Guardrails** → stay in `chat.py` at the edges: input check *before*
  `agent.run()`; output check wraps the streamed event stream. The loop is
  guardrail-agnostic.
- **Tracing** → a span wraps `run()` (the existing `pai_agent_wrapper` concept);
  `dispatch` instruments each tool call.
- **Streaming** → `_stream_turn` yields the same chunk types emitted today, so
  the SSE serializer is unchanged.

### What is removed

- The `pending` / `buffering` / `_looks_like_unfinished_intent` machinery and the
  dual-path "withhold then flush" streaming logic (~80 lines, most of the loop's
  branching). Text streams straight through.
- **Behavior change:** if a weak model emits "让我搜索…" without a tool call, the
  turn ends as plain text — we no longer re-prompt. Accepted.

## Caller changes

- **`agent_service`**: `parse_attachment_tools` returns `(tools, attachments)`;
  it no longer mutates the message. `create_agent` builds the `Agent` and the
  `AgentContext`.
- **`chat.py`**: builds `AgentContext`
  (`history = keep_last_rounds(from_thread(messages[:-1]), N)`, `current_turn`,
  `attachments`, `tools`, `run_vars`), runs the input guardrail before
  `agent.run()`, wraps the event stream with the output guardrail.

### Free win

Because the user message is **never mutated** (attachments are separate context
data; the time-prefix is added inside `build_messages` on the built copy), the
P1 history-pollution bug becomes structurally impossible: no `deepcopy` needed
when saving history, and `_clean_user_message`'s system-time stripping is no
longer necessary.

## Testing

- **`FakeLLM`** yielding scripted chunks makes the loop deterministically
  testable with no network. Cover: text-only-stop, tool-call→text,
  `return_direct`, max-steps.
- Unit tests per unit:
  - `from_thread` normalization (content arrays, assistant tool-call parts, tool
    results, orphaned tool messages).
  - `keep_last_rounds`.
  - **`build_messages`** asserts attachment text actually lands in model input —
    directly locks the bug that started this work.
  - `ToolBox.dispatch` (arg parsing, retry, error capture) and
    `is_return_direct`.
  - `budget.fit` — port the existing `message_manager` tests.

## Migration sequencing

Each step lands green before the next:

1. Add new modules alongside the old (`react_agent.py` still live).
2. Flip `chat.py` / `agent_service` to `Agent` + `AgentContext`.
3. Delete `react_agent.py` + `state.py`; remove the now-dead `deepcopy` /
   `_clean_user_message` system-time handling.

## Risks

- **Token-budgeting port:** `message_manager` is subtle; porting tests first
  protects behavior.
- **Multimodal content:** `Message.content` must still support the list/parts
  shape for vision; text-only stays a plain `str`.
- **Tool-call streaming assembly:** partial `tool_calls` across chunks must be
  reassembled exactly as `react_agent` does today; cover with a `FakeLLM` that
  splits a tool call across chunks.
