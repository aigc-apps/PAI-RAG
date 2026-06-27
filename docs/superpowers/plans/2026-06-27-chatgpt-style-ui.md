# ChatGPT-style Agent UI + Tool Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the agent's reasoning, tool calls + results, and answer in a modern, minimal ChatGPT-style UI (sidebar + centered column + composer), wired end-to-end (backend streams tool results + includes them in history; the reducer captures them).

**Architecture:** Backend (`newbackend`) emits a `response.tool_result` stream event and attaches tool calls/results to conversation history. Frontend (`newfrontend`) gains `ChatMessage.toolCalls`, a reducer that captures `function_call*` + `response.tool_result`, a unified `parseSSE` stream client (dropping the OpenAI SDK), and a ChatGPT-style component set driven by CSS-variable design tokens.

**Tech Stack:** Python/pytest (newbackend); Vite 6 + React 19 + TS + Tailwind v4 + Vitest (newfrontend). Backend tests: `cd newbackend && uv run pytest tests -q`. Frontend: `cd newfrontend && npm test` / `npm run build`.

**Reference spec:** `docs/superpowers/specs/2026-06-27-chatgpt-style-ui-design.md`. Branch: `personal/yfei/agent-core`.

## Global Constraints

- **Backend changes go in `newbackend/`** (the base). `newbackend` tests run via `uv run pytest`.
- **Keep existing behavior** (background/resume/cancel, memory, model selector). Pure-logic changes are TDD'd; UI components get behavior/state tests (not pixel tests).
- **Design tokens are CSS variables** in `index.css`; components reference them — no hardcoded palette drift.
- **`sequence_number` stays monotonic** in the serializer (resume cursors depend on it).
- Run the relevant suite at the end of each task; full frontend `npm run build` must pass for UI tasks.

---

## Key existing contracts (verified)

- **`newbackend/api/protocol/responses_serializer.py`** — `serialize_response_stream`: `nxt()` monotonic seq; `_sse(event)` = `f"data: {event.model_dump_json()}\n\n"`; the `ToolResult` branch (~line 534) calls `asm.on_tool_result(...)` with NO yield. `_Assembler.on_tool_result(call_id, output, error)` appends a `function_call_output` store item `{call_id, output}`.
- **`newbackend/app/conversations_view.py`** — `group_conversation_messages(items, responses)`; per-turn it builds user + assistant message dicts; **skips** `function_call`/`function_call_output` items (the v1 behavior to change). Item shapes: `function_call` content `{call_id,name,arguments}`, `function_call_output` content `{call_id,output}`.
- **`newfrontend/src/stream/reducer.ts`** — `StreamState{message,conversationId?,responseId?,lastSequenceNumber}`; `reduceStreamEvent` switch with a `default` no-op; events typed loosely via `f(event)`.
- **`newfrontend/src/types.ts`** — `ChatMessage{id,role,text,reasoning,reasoningStatus,status,responseId?,previousResponseId?,error?,usage?,lastSequenceNumber?}`; `ConversationDetail.messages`.
- **`newfrontend/src/api/client.ts`** — `streamResponse(params, signal)` uses the OpenAI SDK. **`newfrontend/src/lib/sse.ts`** — `parseSSE(stream)` yields parsed events. **`newfrontend/src/api/responses.ts`** — `streamResume` uses `fetch`+`parseSSE`; `cancelResponse`.
- **`newfrontend/src/store/chat.ts`** — `normalizeHistoryMessages(detail)` maps wire rows → `ChatMessage[]`.
- **Components:** `App, Sidebar, ChatView, MessageList, UserMessage, AssistantMessage, CollapsibleReasoning, Markdown, Composer, MessageControls, ModelSelector` (+ `__tests__`). `cn` in `lib/cn.ts`. lucide-react, sonner, @radix-ui/react-collapsible, react-markdown, remark-gfm, react-syntax-highlighter available.
- **`useResponsesChat`** iterates `streamResponse(...)` through `reduceStreamEvent` → `updateLast(...)`; `updateLast` patches the last message with partial `ChatMessage` fields → it already propagates any new field (e.g. `toolCalls`).

---

## Task 1: Backend — stream tool results

**Files:** Modify `newbackend/api/protocol/responses_serializer.py`. Test: `newbackend/tests/test_tool_result_stream.py`.

**Interfaces:** Produces a `response.tool_result` SSE event `{type, call_id, output, ok, sequence_number}` emitted when a `ToolResult` AgentEvent occurs in the stream. Sync serializer unchanged.

- [ ] **Step 1: Write the failing test** — `newbackend/tests/test_tool_result_stream.py`

```python
import sys, os, asyncio, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.core.events import (RunStarted, ToolStarted, ToolCompleted, ToolResult,
                               TextDelta, RunCompleted, Usage)
from api.protocol.responses_serializer import serialize_response_stream


async def _events():
    yield RunStarted(response_id="resp_1", conversation_id="conv_1")
    yield ToolStarted(call_id="c1", name="web_fetch")
    yield ToolCompleted(call_id="c1", name="web_fetch", arguments='{"url":"http://x"}')
    yield ToolResult(call_id="c1", name="web_fetch", ok=True, output="PAGE TEXT")
    yield TextDelta(text="done")
    yield RunCompleted(usage=Usage(input=1, output=1, total=2))


def _parse(chunks):
    evs = []
    for ln in chunks:
        for part in ln.splitlines():
            if part.startswith("data:"):
                p = part[len("data:"):].strip()
                if p and p != "[DONE]":
                    evs.append(json.loads(p))
    return evs


def test_stream_emits_tool_result_event():
    async def run():
        sink = {}
        chunks = [c async for c in serialize_response_stream(
            _events(), model="m", response_id="resp_1", conversation_id="conv_1", sink=sink)]
        evs = _parse(chunks)
        tr = [e for e in evs if e.get("type") == "response.tool_result"]
        assert len(tr) == 1
        assert tr[0]["call_id"] == "c1" and tr[0]["output"] == "PAGE TEXT" and tr[0]["ok"] is True
        assert isinstance(tr[0]["sequence_number"], int)
        # the function_call args.done event still precedes it
        types = [e["type"] for e in evs]
        assert "response.function_call_arguments.done" in types
        assert types.index("response.function_call_arguments.done") < types.index("response.tool_result")
    asyncio.run(run())
```

- [ ] **Step 2: Run to verify failure** — `cd newbackend && uv run pytest tests/test_tool_result_stream.py -q` → FAIL (no tool_result event).

- [ ] **Step 3: Emit the event** in `serialize_response_stream`

Add a small object-SSE helper near `_sse`:
```python
import json as _json
def _sse_obj(obj: dict) -> str:
    return f"data: {_json.dumps(obj, ensure_ascii=False)}\n\n"
```
In the streaming loop's `ToolResult` branch, after `asm.on_tool_result(ev.call_id, ev.output, ev.error)`, add:
```python
            yield _sse_obj({
                "type": "response.tool_result",
                "call_id": ev.call_id,
                "output": ev.output if ev.output is not None else (ev.error or ""),
                "ok": ev.ok,
                "sequence_number": nxt(),
            })
```
(`ev.ok` exists on `ToolResult`. `nxt()` keeps the cursor monotonic.)

- [ ] **Step 4: Run to verify pass + the existing serializer tests** — `cd newbackend && uv run pytest tests/test_tool_result_stream.py tests/test_responses_serializer_stream.py tests/test_serializer_cancel.py -q` → pass.

- [ ] **Step 5: Commit**

```bash
git add newbackend/api/protocol/responses_serializer.py newbackend/tests/test_tool_result_stream.py
git commit -m "feat(newbackend): stream a response.tool_result event on tool completion"
```

---

## Task 2: Backend — tool calls/results in conversation history

**Files:** Modify `newbackend/app/conversations_view.py`. Test: extend `newbackend/tests/test_conversations_view.py`.

**Interfaces:** `group_conversation_messages` attaches `tool_calls: [{call_id,name,arguments,output}]` to the assistant message dict (matching `function_call` with its `function_call_output` by `call_id`), in order. Assistant messages with no tools get `tool_calls: []`.

- [ ] **Step 1: Write the failing test** — add to `newbackend/tests/test_conversations_view.py`

```python
def test_assistant_message_carries_tool_calls():
    items = _items([
        ("message", "user", {"text": "fetch x"}, "resp_1"),
        ("function_call", None, {"call_id": "c1", "name": "web_fetch", "arguments": "{\"url\":\"x\"}"}, "resp_1"),
        ("function_call_output", None, {"call_id": "c1", "output": "PAGE"}, "resp_1"),
        ("message", "assistant", {"text": "here it is"}, "resp_1"),
    ])
    resps = [StoredResponse(id="resp_1", model="m", status="completed", conversation_id="c")]
    msgs = group_conversation_messages(items, resps)
    assistant = msgs[1]
    assert assistant["role"] == "assistant" and assistant["text"] == "here it is"
    assert assistant["tool_calls"] == [
        {"call_id": "c1", "name": "web_fetch", "arguments": "{\"url\":\"x\"}", "output": "PAGE"}
    ]
    # a tool-less turn still has an empty list
    plain = group_conversation_messages(
        _items([("message", "user", {"text": "hi"}, "r2"),
                ("message", "assistant", {"text": "yo"}, "r2")]),
        [StoredResponse(id="r2", model="m", status="completed", conversation_id="c")])
    assert plain[1]["tool_calls"] == []
```

(`_items` helper already exists in the file; if not, it builds `Item`s with seq from a list of `(type, role, content, response_id)`.)

- [ ] **Step 2: Run to verify failure** — `cd newbackend && uv run pytest tests/test_conversations_view.py -q` → FAIL.

- [ ] **Step 3: Attach tool_calls** in `group_conversation_messages`

Within the per-group assembly, before/while building the assistant dict, collect the group's tool items:
```python
        fcalls = [i for i in group if i.type == "function_call"]
        outputs = {i.content.get("call_id"): i.content.get("output", "")
                   for i in group if i.type == "function_call_output"}
        tool_calls = [
            {
                "call_id": c.content.get("call_id", ""),
                "name": c.content.get("name", ""),
                "arguments": c.content.get("arguments", "") or "",
                "output": outputs.get(c.content.get("call_id"), ""),
            }
            for c in fcalls
        ]
```
Add `"tool_calls": tool_calls` to the assistant message dict (keep emitting the assistant message whenever the group has a response / assistant item / reasoning, as today). The `function_call`/`function_call_output` items are otherwise still not turned into standalone messages.

- [ ] **Step 4: Run to verify pass + full suite** — `cd newbackend && uv run pytest tests -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add newbackend/app/conversations_view.py newbackend/tests/test_conversations_view.py
git commit -m "feat(newbackend): attach tool_calls (call+result) to assistant history messages"
```

---

## Task 3: Frontend — reducer + types capture tool calls/results

**Files:** Modify `newfrontend/src/types.ts`, `newfrontend/src/stream/reducer.ts`, `newfrontend/src/store/chat.ts`. Test: extend `newfrontend/src/stream/__tests__/reducer.test.ts`, `newfrontend/src/store/__tests__/chat.test.ts`.

**Interfaces:** `ToolUse` type; `ChatMessage.toolCalls: ToolUse[]`; reducer handles `function_call` add/args/done + `response.tool_result`; `normalizeHistoryMessages` maps `tool_calls`.

- [ ] **Step 1: Write the failing tests** — append to `reducer.test.ts`

```ts
describe("reduceStreamEvent — tools", () => {
  it("captures a tool call and its result", () => {
    let s = initialStreamState("tmp");
    s = reduceStreamEvent(s, { type: "response.created", response: { id: "r1", conversation: { id: "c1" } } } as any);
    s = reduceStreamEvent(s, { type: "response.output_item.added", item: { id: "fc_c1", type: "function_call", name: "web_fetch", call_id: "c1" } } as any);
    s = reduceStreamEvent(s, { type: "response.function_call_arguments.delta", item_id: "fc_c1", delta: '{"url":' } as any);
    s = reduceStreamEvent(s, { type: "response.function_call_arguments.done", item_id: "fc_c1", name: "web_fetch", arguments: '{"url":"x"}' } as any);
    s = reduceStreamEvent(s, { type: "response.tool_result", call_id: "c1", output: "PAGE", ok: true } as any);
    s = reduceStreamEvent(s, { type: "response.output_text.delta", delta: "done" } as any);
    expect(s.message.toolCalls).toHaveLength(1);
    const t = s.message.toolCalls[0];
    expect(t.name).toBe("web_fetch");
    expect(t.arguments).toBe('{"url":"x"}');
    expect(t.output).toBe("PAGE");
    expect(t.status).toBe("done");
    expect(s.message.text).toBe("done");
  });

  it("marks a failed tool result as error", () => {
    let s = initialStreamState("tmp");
    s = reduceStreamEvent(s, { type: "response.output_item.added", item: { id: "fc_c1", type: "function_call", name: "web_fetch", call_id: "c1" } } as any);
    s = reduceStreamEvent(s, { type: "response.tool_result", call_id: "c1", output: "boom", ok: false } as any);
    expect(s.message.toolCalls[0].status).toBe("error");
    expect(s.message.toolCalls[0].error).toBe("boom");
  });

  it("initial message has an empty toolCalls array", () => {
    expect(initialStreamState("x").message.toolCalls).toEqual([]);
  });
});
```

And to `chat.test.ts` (normalize):
```ts
it("normalizeHistoryMessages maps tool_calls", () => {
  const out = normalizeHistoryMessages({
    id: "c", title: null, created_at: null, updated_at: null, latest_response_id: "r1",
    messages: [{ role: "assistant", text: "a", reasoning: "", response_id: "r1",
      previous_response_id: null, status: "completed",
      tool_calls: [{ call_id: "c1", name: "web_fetch", arguments: "{}", output: "PAGE" }] } as never],
  });
  expect(out[0].toolCalls).toEqual([
    { id: "c1", name: "web_fetch", arguments: "{}", status: "done", output: "PAGE" },
  ]);
});
```

- [ ] **Step 2: Run to verify failure** — `cd newfrontend && npm test -- "reducer|chat"` → FAIL.

- [ ] **Step 3: Add the type** in `src/types.ts`

```ts
export interface ToolUse {
  id: string;
  name: string;
  arguments: string;
  status: "running" | "done" | "error";
  output?: string;
  error?: string;
}
```
Add to `ChatMessage`: `toolCalls: ToolUse[];` (required; reducer/normalize always set it).

- [ ] **Step 4: Reducer** in `src/stream/reducer.ts`

`initialStreamState`: add `toolCalls: []` to the message. A `call_id` helper (strip an `fc_` prefix from `item_id`):
```ts
function callIdOf(e: Record<string, unknown>): string {
  const item = e.item as { call_id?: string; id?: string } | undefined;
  const raw = (e.call_id as string) || item?.call_id || (e.item_id as string) || item?.id || "";
  return raw.startsWith("fc_") ? raw.slice(3) : raw;
}
```
Add cases (inside `reduceCore`, before `default`):
```ts
    case "response.output_item.added": {
      const item = e.item as { type?: string; name?: string } | undefined;
      if (item?.type === "function_call") {
        const id = callIdOf(e);
        return { ...state, message: { ...msg, toolCalls: [...msg.toolCalls,
          { id, name: item.name || "", arguments: "", status: "running" }] } };
      }
      return state;
    }
    case "response.function_call_arguments.delta": {
      const id = callIdOf(e);
      return { ...state, message: { ...msg, toolCalls: msg.toolCalls.map(t =>
        t.id === id ? { ...t, arguments: t.arguments + String(e.delta ?? "") } : t) } };
    }
    case "response.function_call_arguments.done": {
      const id = callIdOf(e);
      return { ...state, message: { ...msg, toolCalls: msg.toolCalls.map(t =>
        t.id === id ? { ...t, name: (e.name as string) || t.name,
          arguments: String(e.arguments ?? t.arguments) } : t) } };
    }
    case "response.tool_result": {
      const id = callIdOf(e);
      const ok = e.ok !== false;
      return { ...state, message: { ...msg, toolCalls: msg.toolCalls.map(t =>
        t.id === id ? { ...t, status: ok ? "done" : "error",
          output: ok ? String(e.output ?? "") : t.output,
          error: ok ? t.error : String(e.output ?? e.error ?? "") } : t) } };
    }
```
NOTE: the existing `response.output_item.done` case (reasoning) stays; if `item.type==="function_call"` it can fall through unchanged (status stays running until the result event). Keep the `lastSequenceNumber` wrapper as-is — `toolCalls` updates flow through it.

- [ ] **Step 5: History** in `src/store/chat.ts normalizeHistoryMessages`

Map `r.tool_calls` (if present) → `ToolUse[]`:
```ts
    toolCalls: (r.tool_calls ?? []).map((tc: any) => ({
      id: tc.call_id, name: tc.name, arguments: tc.arguments ?? "",
      status: "done" as const, output: tc.output ?? "",
    })),
```
Add `tool_calls?` to the `WireHistoryMessage` interface. Ensure user rows get `toolCalls: []`.

- [ ] **Step 6: Run to verify pass + build** — `cd newfrontend && npm test -- "reducer|chat" && npm run build` → pass. (Other tests that construct `ChatMessage` literals may need `toolCalls: []` added — fix any TS errors by adding the field.)

- [ ] **Step 7: Commit**

```bash
git add newfrontend/src/types.ts newfrontend/src/stream/reducer.ts newfrontend/src/store/chat.ts newfrontend/src/stream/__tests__/reducer.test.ts newfrontend/src/store/__tests__/chat.test.ts
git commit -m "feat(newfrontend): capture tool calls + results into ChatMessage.toolCalls (live + history)"
```

---

## Task 4: Frontend — unified SSE stream client (drop OpenAI SDK)

**Files:** Modify `newfrontend/src/api/client.ts`, `newfrontend/src/stream/reducer.ts` (event type import), `newfrontend/package.json` (drop `openai`). Test: `newfrontend/src/api/__tests__/client.test.ts`.

**Interfaces:** `streamResponse(params, signal)` POSTs to `/v1/responses` with `{...params, store:true, stream:true}` and yields parsed events via `parseSSE`. `ResponseStreamParams` unchanged (incl. `background`).

- [ ] **Step 1: Write the failing test** — `newfrontend/src/api/__tests__/client.test.ts`

```ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { streamResponse } from "../client";

function streamFrom(parts: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({ start(c) { for (const p of parts) c.enqueue(enc.encode(p)); c.close(); } });
}

describe("streamResponse", () => {
  beforeEach(() => vi.restoreAllMocks());
  it("POSTs /v1/responses with stream+background and yields parsed events", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, status: 200,
      body: streamFrom(['data: {"type":"response.created","response":{"id":"r1"}}\n\n',
                        'data: {"type":"response.completed","response":{"id":"r1","status":"completed"}}\n\n']),
    } as unknown as Response);
    vi.stubGlobal("fetch", fetchMock);
    const out: any[] = [];
    for await (const e of streamResponse(
      { model: "m", input: "hi", user_id: "u1", background: true } as any,
      new AbortController().signal)) out.push(e);
    expect(out.map((e) => e.type)).toEqual(["response.created", "response.completed"]);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/v1/responses");
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body);
    expect(body.stream).toBe(true) && expect(body.store).toBe(true) && expect(body.background).toBe(true);
  });
});
```

- [ ] **Step 2: Run to verify failure** — `cd newfrontend && npm test -- client` → FAIL (current client uses the SDK / no such behavior).

- [ ] **Step 3: Rewrite `src/api/client.ts`**

```ts
import { parseSSE } from "../lib/sse";

export interface ResponseStreamParams {
  model: string;
  input: string;
  user_id: string;
  conversation?: string;
  previous_response_id?: string;
  background?: boolean;
}

export function streamResponse(
  params: ResponseStreamParams,
  signal: AbortSignal
): AsyncIterable<unknown> {
  return (async function* () {
    const res = await fetch("/v1/responses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...params, store: true, stream: true }),
      signal,
    });
    if (!res.ok || !res.body) throw new Error(`stream failed: ${res.status}`);
    yield* parseSSE(res.body);
  })();
}
```

- [ ] **Step 4: Drop the SDK type in the reducer** — in `src/stream/reducer.ts`, replace `import type { ResponseStreamEvent } from "openai/...";` and the param type with a local loose type:
```ts
type StreamEvent = Record<string, unknown>;
```
and change `reduceStreamEvent(state: StreamState, event: StreamEvent)` / `f(event: StreamEvent)` accordingly. Remove the `openai` import.

- [ ] **Step 5: Remove `openai` from deps** — `npm uninstall openai` (updates package.json + lock). Grep to confirm no remaining `from "openai"` imports anywhere in `src/`.

- [ ] **Step 6: Run to verify pass + full suite + build** — `cd newfrontend && npm test && npm run build` → pass (the hook test mocks `../../api/client` so it's unaffected; the reducer tests feed plain objects).

- [ ] **Step 7: Commit**

```bash
git add newfrontend/src/api/client.ts newfrontend/src/stream/reducer.ts newfrontend/package.json newfrontend/package-lock.json newfrontend/src/api/__tests__/client.test.ts
git commit -m "feat(newfrontend): unified SSE stream client (parseSSE for POST), drop openai SDK"
```

---

## Task 5: Frontend — design system + app shell + sidebar + composer

**Files:** Modify `newfrontend/src/index.css`, `App.tsx`, `Sidebar.tsx`, `ChatView.tsx`, `Composer.tsx`, `ModelSelector.tsx`. Test: update `App.test.tsx` (smoke).

**Interfaces:** No prop/behavior changes (send/stop/regenerate/resume, model selection, new chat/select/delete unchanged) — purely visual + layout. Empty-state composer when no messages.

- [ ] **Step 1: Write the design tokens** — replace `src/index.css`

```css
@import "tailwindcss";

:root {
  --bg: #ffffff;
  --bg-sidebar: #f9f9f9;
  --bg-elevated: #ffffff;
  --text: #0d0d0d;
  --text-muted: #676767;
  --text-faint: #9b9b9b;
  --border: #e5e5e5;
  --border-strong: #d9d9d9;
  --user-bubble: #f4f4f4;
  --tool-bg: #f7f7f8;
  --accent: #0d0d0d;
  --accent-fg: #ffffff;
  --danger: #e02e2e;
  --radius: 12px;
}

html, body, #root { height: 100%; margin: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
    Arial, "Apple Color Emoji", "Segoe UI Emoji", sans-serif;
  font-size: 16px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
* { box-sizing: border-box; }
.scrollbar-thin::-webkit-scrollbar { width: 8px; }
.scrollbar-thin::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 8px; }
```

- [ ] **Step 2: Build the shell + components to this spec** (write the JSX; behavior is unchanged from the current components — restyle only). Requirements:

  **App.tsx:** `<div class="flex h-full">` Sidebar + a `<main class="flex flex-1 flex-col min-w-0">` containing `<ChatView/>` and a `<Toaster/>` (Toaster already in main.tsx). A sidebar open/close state lives in App or a small zustand UI flag; render a hamburger toggle in the top bar that hides/shows the sidebar (CSS width/translate). Keep it simple: a `useState(true)`.

  **Sidebar.tsx:** width `w-[260px]`, `bg-[var(--bg-sidebar)]`, full height, `flex flex-col`, right border. Top: a "New chat" button — full-width, `rounded-lg`, ghost style, lucide `Plus` + "New chat", hover bg. List: `flex-1 overflow-y-auto scrollbar-thin`, each item a row `rounded-lg px-3 py-2 text-sm truncate` with hover bg and active bg (`--user-bubble`) when `selectedId===id`; a lucide `Trash2` button that appears on hover (`group-hover:visible`) calling delete (stopPropagation). Behavior identical to the current Sidebar (refresh on mount, openConversation, newChat, onDelete). When the sidebar is hidden, App shifts it out.

  **ChatView.tsx (+ top bar):** `flex h-full flex-col`. Top bar: a thin row (`h-12 border-b border-[var(--border)] flex items-center px-3 gap-2`) with the sidebar toggle (lucide `PanelLeft`), the `ModelSelector`, and flex spacer. Then `<MessageList/>` (flex-1, scroll). Then the `Composer` area. **Empty state:** if `messages.length===0`, instead of the list+bottom composer, center a column: a heading ("What can I help with?") and the Composer beneath it, vertically centered (`flex-1 flex flex-col items-center justify-center`). Keep `resumeIfInterrupted` wiring (the useEffect) as-is.

  **Composer.tsx:** centered `w-full max-w-3xl mx-auto`; a container `rounded-3xl border border-[var(--border-strong)] bg-[var(--bg-elevated)] shadow-sm px-3 py-2 flex items-end gap-2`; an auto-grow `<textarea>` (rows=1, grows to ~6 lines, no resize, no border, `outline-none bg-transparent`, placeholder "Message the agent…"); a circular button on the right: when `isStreaming` a square stop (lucide `Square`) calling `onStop`, else an up-arrow (lucide `ArrowUp`) in a `rounded-full bg-[var(--accent)] text-[var(--accent-fg)] h-8 w-8` disabled when empty, calling submit. Enter submits (not Shift+Enter); clears on submit. Add aria-labels ("Send"/"Stop"). A small footer line under the composer is optional.

  **ModelSelector.tsx:** keep the `/v1/models` fetch behavior; restyle the `<select>` as a clean, borderless-on-hover dropdown (`text-sm font-medium text-[var(--text)] rounded-lg px-2 py-1 hover:bg-[var(--user-bubble)]`).

- [ ] **Step 3: Update `App.test.tsx`** — it mocks `../../api/conversations`, `../../api/client`, `../../lib/user`, `../../api/models`. Keep those mocks. Assert: the composer textbox renders; the "New chat" button renders (findByRole, async sidebar refresh); and seeding `useChatStore` with a message renders its text. (The empty-state path: with no messages, the composer still renders — adjust the assertion to `getByRole("textbox")` which holds in both states.)

- [ ] **Step 4: Run to verify pass + build** — `cd newfrontend && npm test -- App && npm run build` → pass.

- [ ] **Step 5: Commit**

```bash
git add newfrontend/src/index.css newfrontend/src/components/App.tsx newfrontend/src/components/Sidebar.tsx newfrontend/src/components/ChatView.tsx newfrontend/src/components/Composer.tsx newfrontend/src/components/ModelSelector.tsx newfrontend/src/components/__tests__/App.test.tsx
git commit -m "feat(newfrontend): ChatGPT-style shell — sidebar, top bar, composer, empty state, design tokens"
```

---

## Task 6: Frontend — message timeline: reasoning, tool cards, answer

**Files:** Modify `MessageList.tsx`, `UserMessage.tsx`, `AssistantMessage.tsx`, `CollapsibleReasoning.tsx`, `Markdown.tsx`, `MessageControls.tsx`; create `ToolCall.tsx`. Tests: `ToolCall.test.tsx` (new), update `AssistantMessage.test.tsx`, `CollapsibleReasoning.test.tsx`.

**Interfaces:** `ToolCall({ tool }: { tool: ToolUse })`; `AssistantMessage` renders `CollapsibleReasoning` → `tool` cards (from `message.toolCalls`) → answer/error/cancelled → controls.

- [ ] **Step 1: Write the failing tests**

`ToolCall.test.tsx`:
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ToolCall } from "../ToolCall";

describe("ToolCall", () => {
  it("shows name + result and expands details", async () => {
    render(<ToolCall tool={{ id: "c1", name: "web_fetch", arguments: '{"url":"x"}', status: "done", output: "PAGE TEXT" }} />);
    expect(screen.getByText("web_fetch")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button"));
    expect(screen.getByText(/PAGE TEXT/)).toBeInTheDocument();
    expect(screen.getByText(/"url"/)).toBeInTheDocument();
  });
  it("shows an error status", () => {
    render(<ToolCall tool={{ id: "c1", name: "web_fetch", arguments: "{}", status: "error", error: "boom" }} />);
    expect(screen.getByText("web_fetch")).toBeInTheDocument();
    expect(screen.getByText(/error/i)).toBeInTheDocument();
  });
});
```

Update `AssistantMessage.test.tsx` — add (keep `msg()` helper, add `toolCalls: []` to its defaults):
```tsx
it("renders tool cards from toolCalls", () => {
  render(<AssistantMessage message={msg({ status: "completed", text: "ans",
    toolCalls: [{ id: "c1", name: "web_fetch", arguments: "{}", status: "done", output: "PAGE" }] })} />);
  expect(screen.getByText("web_fetch")).toBeInTheDocument();
  expect(screen.getByText("ans")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run to verify failure** — `cd newfrontend && npm test -- "ToolCall|AssistantMessage"` → FAIL (no ToolCall; toolCalls not rendered).

- [ ] **Step 3: Implement `ToolCall.tsx`**

A bordered card using Radix Collapsible: header row = a small status glyph (running → lucide `Loader2` w/ `animate-spin`; done → `Check`; error → `X`), a lucide `Wrench`, the tool `name` (mono/medium), a muted status word; clicking toggles details. Details = the `arguments` (mono, `text-xs`, `whitespace-pre-wrap break-all`, label "Arguments") and the `output`/`error` (mono, `text-xs`, scrollable `max-h-64 overflow-auto`, label "Result"/"Error", truncate-friendly). Card classes: `rounded-xl border border-[var(--border)] bg-[var(--tool-bg)] text-sm my-2`. Whole header is the `<button>` (so the tests' `getByRole("button")` works).

- [ ] **Step 4: Update components to the timeline**

  **AssistantMessage.tsx:** order = `CollapsibleReasoning(reasoning,status)` → `message.toolCalls.map(t => <ToolCall key=t.id tool=t/>)` → then: `status==="failed"` → error box; else `<Markdown content={message.text}/>` (render only if text non-empty); then `status==="stopped"`→"stopped" note, `status==="cancelled"`→"cancelled" note; then `MessageControls` when `completed||cancelled`. Wrap in a left-aligned full-width block with a small assistant glyph; `max-w-3xl mx-auto` is applied by MessageList, so AssistantMessage is just the block.

  **MessageList.tsx:** center column `mx-auto w-full max-w-3xl px-4 py-6 space-y-6`; map messages → `UserMessage`/`AssistantMessage`; pass `onRegenerate` only to the last assistant; keep autoscroll.

  **UserMessage.tsx:** right-aligned `flex justify-end`; bubble `bg-[var(--user-bubble)] rounded-3xl px-4 py-2.5 max-w-[80%] whitespace-pre-wrap`.

  **CollapsibleReasoning.tsx:** restyle as a "Thinking" disclosure — a muted button row (chevron rotates; label "Thinking…" while `status==="streaming"`, "Thought" when "done") + collapsible body `text-sm text-[var(--text-muted)] whitespace-pre-wrap border-l-2 border-[var(--border)] pl-3`. Keep the auto-open-while-streaming / auto-collapse-on-done behavior + the "render nothing when reasoning empty" rule.

  **Markdown.tsx:** keep react-markdown + remark-gfm + code highlighting + inline-code styling; add a `.prose`-ish typographic wrapper using tokens (headings weight/size, list spacing, `a` underline, `blockquote` border, `table` borders). Keep a copy button on fenced code blocks.

  **MessageControls.tsx:** restyle the copy/regenerate as subtle ghost icon buttons (`text-[var(--text-faint)] hover:text-[var(--text)]`), shown on message hover.

- [ ] **Step 5: Run to verify pass + full suite + build** — `cd newfrontend && npm test && npm run build` → pass.

- [ ] **Step 6: Commit**

```bash
git add newfrontend/src/components/ToolCall.tsx newfrontend/src/components/MessageList.tsx newfrontend/src/components/UserMessage.tsx newfrontend/src/components/AssistantMessage.tsx newfrontend/src/components/CollapsibleReasoning.tsx newfrontend/src/components/Markdown.tsx newfrontend/src/components/MessageControls.tsx newfrontend/src/components/__tests__
git commit -m "feat(newfrontend): assistant timeline — Thinking disclosure, tool cards, styled Markdown answer"
```

---

## Task 7: Polish + manual smoke

**Files:** any small fixes. Test: full suites + build + a manual run.

- [ ] **Step 1: Full frontend suite + build** — `cd newfrontend && npm test && npm run build` → all green.
- [ ] **Step 2: Full backend suite** — `cd newbackend && uv run pytest tests -q` → all green.
- [ ] **Step 3: Manual smoke (optional, needs the service):** `cd newbackend && uv run uvicorn app.lean_main:app --port 8000` (configure a model + `MEMORY_ENABLED`/tools as desired), then `cd newfrontend && npm run dev`. Verify: empty-state greeting; sending a message streams the answer; reasoning shows as a collapsible "Thinking" that collapses when done; a tool-using turn (e.g. ask for the current time or to fetch a URL) shows a tool card with status + expandable args/result; the sidebar lists/loads/deletes conversations; stop/regenerate/copy work; reloading a conversation shows the same reasoning/tools/answer.
- [ ] **Step 4: Commit any fixes**

```bash
git add -A newfrontend newbackend
git commit -m "chore(newfrontend): polish + verify ChatGPT-style UI end-to-end"
```

---

## Self-Review (against the spec)

- **Tools shown correctly end-to-end:** backend streams `response.tool_result` (Task 1) + history grouping attaches `tool_calls` (Task 2); reducer captures calls+results live and from history (Task 3); `ToolCall` cards render them (Task 6).
- **Reasoning + answer:** Thinking disclosure + Markdown answer (Task 6), unchanged data path.
- **ChatGPT-style layout:** design tokens + shell + sidebar + composer + empty state (Task 5); centered column + timeline (Task 6).
- **Conversation list:** restyled Sidebar (Task 5), behavior unchanged.
- **Unified SSE client** enables the custom tool-result event and drops the OpenAI SDK (Task 4); background/resume/cancel/memory/model-selector preserved.
- **Tests:** logic TDD'd (serializer, grouping, reducer, normalize, client); components behavior-tested (ToolCall, AssistantMessage, CollapsibleReasoning, Composer, App). Visual quality is the user's call to iterate.

No placeholders in logic tasks (full code). UI tasks specify exact tokens, structure, behavior, and tests; the implementer writes idiomatic JSX to those contracts.
