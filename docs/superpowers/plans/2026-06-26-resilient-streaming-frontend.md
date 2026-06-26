# Resilient Streaming — Frontend Reconnect & Cancel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `newfrontend/` resilient: chat turns run in `background` mode, a same-session interruption (tab backgrounded / laptop sleep / network drop) reconnects and resumes the in-flight answer from a sequence cursor, and Stop becomes a real server-side cancel that persists a continuable partial.

**Architecture:** The pure reducer gains a `lastSequenceNumber` cursor and a `response.incomplete`→`cancelled` terminal. A tiny SSE reader (`lib/sse.ts`) + a `responses` API client power `streamResume`/`cancelResponse`. `useResponsesChat` sends `background:true`, drives both the initial stream and a resume stream through one shared consume loop, fires server-side cancel on Stop (deferring until the `response_id` is known), and advances conversation anchors on `completed` **or** `cancelled`. The shell reconnects on `visibilitychange`/`online`.

**Tech Stack:** Vite 6, React 19, TypeScript 5 (strict), Vitest 3 + `@testing-library/react` + jsdom. The OpenAI JS SDK is used for the initial POST; resume uses a plain `fetch`+SSE reader. All commands run from `newfrontend/`.

**Reference spec:** `docs/superpowers/specs/2026-06-26-resilient-streaming-design.md`. **Prerequisite plan:** `2026-06-26-resilient-streaming-backend.md` (the `background` flag, `GET ?stream=&starting_after=`, and `POST .../cancel` endpoints) must be implemented first. Branch: `personal/yfei/agent-core`.

## Global Constraints

- **Isolated to `newfrontend/`.** Do not touch `frontend/` or `backend/`. TypeScript strict mode (`noUnusedLocals`/`noUnusedParameters` are on — no unused params).
- **The reducer (`stream/reducer.ts`) stays pure** — `(state, event) => state`, new objects, no I/O. It is the conformance gate to the backend serializer.
- **Resume cursor = `sequence_number`.** Track the max `sequence_number` seen; resume requests `?starting_after=<that>`; the backend returns only events with `sequence_number > N`. The reducer must capture `sequence_number` on **every** event (including otherwise-ignored ones) so the cursor never lags.
- **Cancel wire shape:** the terminal for a cancelled run is `type:"response.incomplete"` whose embedded `response.status === "cancelled"`. The reducer keys final status off `response.status`, not the event type.
- **A cancelled turn is a real, continuable response:** advance `conversationId`/`lastResponseId` on `cancelled` exactly as on `completed`.
- **Scope:** live resume covers **same-session** interruptions (store state survives). Hard-reload live re-attach is out of scope (no data is lost — the run finishes server-side and the turn appears on next conversation load).
- `npm run build` (tsc + vite) and `npm test` (vitest run) must both pass at the end of every task.

---

## Key existing code (verified, do not re-derive)

- **`src/types.ts`** — `MessageStatus = "streaming" | "completed" | "failed" | "stopped"`; `ChatMessage { id, role, text, reasoning, reasoningStatus, status, responseId?, previousResponseId?, error?, usage? }`. (This plan adds `"cancelled"` and `lastSequenceNumber?`.)
- **`src/stream/reducer.ts`** — `StreamState { message, conversationId?, responseId? }`; `initialStreamState(id)`; `reduceStreamEvent(state, event)` switching on `e.type` with a `f(event)` cast helper. Handles `response.created`/`output_text.delta`/`reasoning_summary_text.delta`/`reasoning_summary_part.done`/`reasoning_summary_text.done`/`output_item.done`(reasoning)/`completed`/`failed`; `default` returns state.
- **`src/api/client.ts`** — `streamResponse(params: ResponseStreamParams, signal): AsyncIterable<ResponseStreamEvent>` calling `client.responses.create({...params, store:true, stream:true} as never, {signal})`. `ResponseStreamParams { model, input, user_id, conversation?, previous_response_id? }`.
- **`src/api/conversations.ts`** — `listConversations`/`getConversation`/`deleteConversation`; a `jsonOrThrow` helper pattern; uses `fetch` + `encodeURIComponent`.
- **`src/store/chat.ts`** — `useChatStore` with `messages`, `status`, `model`, `conversationId`, `lastResponseId` and actions `appendMessage`, `updateLast(patch: Partial<ChatMessage>)` (patches the last message), `setStatus`, `setModel`, `setAnchors({conversationId?, lastResponseId?})`, `loadHistory`, `reset`.
- **`src/store/conversations.ts`** — `useConversationsStore` with `refresh()`.
- **`src/hooks/useResponsesChat.ts`** — current shape: `{ send, stop, regenerate, isStreaming }`; `runTurn` appends a user + a streaming assistant message, opens an `AbortController`, iterates `streamResponse` through `reduceStreamEvent`+`updateLast`, and (v1) `stop` aborts locally. **This plan rewrites the hook.**
- **`src/components/ChatView.tsx`** — renders header + `ModelSelector` + `MessageList` + `Composer`; calls `useResponsesChat()`.
- **`src/components/AssistantMessage.tsx`** — renders `CollapsibleReasoning`, then (failed → error box | `Markdown`), a `stopped` note, and `MessageControls` when `completed`.
- **Test conventions:** vitest globals on; tests import `{ describe, it, expect, vi, beforeEach }` from `"vitest"` and use `@testing-library/react` (`renderHook`, `render`, `act`). Module mocks via `vi.mock("../../path", () => ({...}))`. `vitest.setup.ts` already polyfills `scrollIntoView`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/types.ts` (modify) | `MessageStatus` += `"cancelled"`; `ChatMessage` += `lastSequenceNumber?: number` |
| `src/stream/reducer.ts` (modify) | `StreamState.lastSequenceNumber`; capture `sequence_number`; `response.incomplete`→`cancelled` |
| `src/lib/sse.ts` (new) | `parseSSE(stream)` — chunked SSE → parsed event objects |
| `src/api/responses.ts` (new) | `cancelResponse(id)`, `streamResume(id, startingAfter, signal)` |
| `src/hooks/useResponsesChat.ts` (rewrite) | background send; shared consume loop; Stop→cancel; `resumeIfInterrupted` |
| `src/components/ChatView.tsx` (modify) | wire `resumeIfInterrupted` to `visibilitychange`/`online`/mount |
| `src/components/AssistantMessage.tsx` (modify) | render the `cancelled` state |
| `src/stream/__tests__/reducer.test.ts` (modify) | add seq + incomplete tests |
| `src/lib/__tests__/sse.test.ts` (new) | SSE parser |
| `src/api/__tests__/responses.test.ts` (new) | cancel + resume clients |
| `src/hooks/__tests__/useResponsesChat.test.tsx` (rewrite) | background/cancel/resume |
| `src/components/__tests__/AssistantMessage.test.tsx` (new) | cancelled rendering |

---

## Task 1: Reducer — sequence cursor + cancelled terminal

Add `lastSequenceNumber` tracking and a `response.incomplete`→`cancelled` case, keeping the reducer pure. Existing reducer tests stay green.

**Files:**
- Modify: `src/types.ts`, `src/stream/reducer.ts`
- Test: `src/stream/__tests__/reducer.test.ts` (append cases)

**Interfaces:**
- Produces: `MessageStatus` includes `"cancelled"`; `ChatMessage.lastSequenceNumber?: number`; `StreamState { message, conversationId?, responseId?, lastSequenceNumber: number }`; `initialStreamState(id)` sets `lastSequenceNumber: 0`; `reduceStreamEvent` captures `event.sequence_number` (monotonic max) on every event and maps `response.incomplete` (status `cancelled`/`incomplete`) → message `status:"cancelled"`.

- [ ] **Step 1: Write the failing test** — append to `src/stream/__tests__/reducer.test.ts`

```ts
// (append below the existing tests; `fold`, `created`, `textDelta` already exist)
import { describe as _d } from "vitest"; // no-op import guard if needed

describe("reduceStreamEvent — resilience", () => {
  const withSeq = (ev: any, n: number) => ({ ...ev, sequence_number: n });

  it("captures the max sequence_number across events", () => {
    let s = initialStreamState("tmp");
    s = reduceStreamEvent(s, withSeq({ type: "response.created", response: { id: "r", conversation: { id: "c" } } }, 1) as any);
    s = reduceStreamEvent(s, withSeq(textDelta("hi"), 5) as any);
    s = reduceStreamEvent(s, withSeq({ type: "response.in_progress" }, 3) as any); // older/ignored
    expect(s.lastSequenceNumber).toBe(5);
    expect(s.message.lastSequenceNumber).toBe(5);
  });

  it("maps response.incomplete (status cancelled) to a cancelled message", () => {
    let s = initialStreamState("tmp");
    s = reduceStreamEvent(s, { type: "response.created", response: { id: "r1", conversation: { id: "c1" } } } as any);
    s = reduceStreamEvent(s, textDelta("partial") as any);
    s = reduceStreamEvent(s, {
      type: "response.incomplete",
      response: { id: "r1", conversation: { id: "c1" }, status: "cancelled" },
    } as any);
    expect(s.message.status).toBe("cancelled");
    expect(s.message.text).toBe("partial");
    expect(s.responseId).toBe("r1");
    expect(s.conversationId).toBe("c1");
  });

  it("initial state starts at sequence 0", () => {
    expect(initialStreamState("x").lastSequenceNumber).toBe(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd newfrontend && npm test -- reducer`
Expected: FAIL — `lastSequenceNumber` is `undefined`; `response.incomplete` falls through to `default` so status stays `streaming`.

- [ ] **Step 3: Update `src/types.ts`**

Change `MessageStatus` and add the field to `ChatMessage`:

```ts
export type MessageStatus =
  | "streaming"
  | "completed"
  | "failed"
  | "stopped"
  | "cancelled";
```

In `ChatMessage`, add after `usage?`:

```ts
  /** highest response.* sequence_number folded so far (resume cursor) */
  lastSequenceNumber?: number;
```

- [ ] **Step 4: Rewrite `src/stream/reducer.ts`**

Replace the whole file with:

```ts
import type { ResponseStreamEvent } from "openai/resources/responses/responses";
import type { ChatMessage } from "../types";

export interface StreamState {
  message: ChatMessage;
  conversationId?: string;
  responseId?: string;
  lastSequenceNumber: number;
}

export function initialStreamState(id: string): StreamState {
  return {
    message: {
      id,
      role: "assistant",
      text: "",
      reasoning: "",
      reasoningStatus: "idle",
      status: "streaming",
      lastSequenceNumber: 0,
    },
    lastSequenceNumber: 0,
  };
}

// Narrow helper: read a field off a loosely-typed event without fighting the
// SDK's giant discriminated union in every branch.
function f(event: ResponseStreamEvent): Record<string, unknown> {
  return event as unknown as Record<string, unknown>;
}

function reduceCore(state: StreamState, event: ResponseStreamEvent): StreamState {
  const e = f(event);
  const msg = state.message;

  switch (e.type) {
    case "response.created": {
      const response = e.response as
        | { id?: string; conversation?: { id?: string } | null }
        | undefined;
      const responseId = response?.id;
      const conversationId = response?.conversation?.id;
      return {
        ...state,
        responseId: responseId ?? state.responseId,
        conversationId: conversationId ?? state.conversationId,
        message: {
          ...msg,
          responseId: responseId ?? msg.responseId,
          id: responseId ?? msg.id,
        },
      };
    }

    case "response.output_text.delta": {
      return {
        ...state,
        message: { ...msg, text: msg.text + String(e.delta ?? "") },
      };
    }

    case "response.reasoning_summary_text.delta": {
      return {
        ...state,
        message: {
          ...msg,
          reasoning: msg.reasoning + String(e.delta ?? ""),
          reasoningStatus: "streaming",
        },
      };
    }

    case "response.reasoning_summary_part.done":
    case "response.reasoning_summary_text.done": {
      return { ...state, message: { ...msg, reasoningStatus: "done" } };
    }

    case "response.output_item.done": {
      const item = e.item as { type?: string } | undefined;
      if (item?.type === "reasoning" && msg.reasoningStatus !== "idle") {
        return { ...state, message: { ...msg, reasoningStatus: "done" } };
      }
      return state;
    }

    case "response.completed":
    case "response.incomplete": {
      const response = e.response as
        | {
            id?: string;
            conversation?: { id?: string } | null;
            status?: string;
            usage?: {
              input_tokens?: number;
              output_tokens?: number;
              total_tokens?: number;
            } | null;
          }
        | undefined;
      const usage = response?.usage
        ? {
            input: response.usage.input_tokens ?? 0,
            output: response.usage.output_tokens ?? 0,
            total: response.usage.total_tokens ?? 0,
          }
        : msg.usage;
      // response.incomplete carries status "cancelled" (server-side cancel) — and
      // tolerates "incomplete" (length cap, out of scope) by treating it the same.
      const status: ChatMessage["status"] =
        e.type === "response.completed"
          ? "completed"
          : response?.status === "incomplete"
          ? "cancelled"
          : (response?.status as ChatMessage["status"]) ?? "cancelled";
      return {
        ...state,
        responseId: response?.id ?? state.responseId,
        conversationId: response?.conversation?.id ?? state.conversationId,
        message: {
          ...msg,
          status,
          reasoningStatus:
            msg.reasoningStatus === "streaming" ? "done" : msg.reasoningStatus,
          responseId: response?.id ?? msg.responseId,
          usage,
        },
      };
    }

    case "response.failed": {
      const response = e.response as
        | { error?: { message?: string } | null }
        | undefined;
      return {
        ...state,
        message: {
          ...msg,
          status: "failed",
          error: response?.error?.message ?? "request failed",
        },
      };
    }

    default:
      // response.in_progress, output_item.added, content_part.*, *_text.done,
      // function_call* — no effect on the rendered message.
      return state;
  }
}

export function reduceStreamEvent(
  state: StreamState,
  event: ResponseStreamEvent
): StreamState {
  const e = f(event);
  const next = reduceCore(state, event);
  const seq =
    typeof e.sequence_number === "number"
      ? Math.max(state.lastSequenceNumber, e.sequence_number as number)
      : state.lastSequenceNumber;
  if (next === state && seq === state.lastSequenceNumber) return state;
  return {
    ...next,
    lastSequenceNumber: seq,
    message: { ...next.message, lastSequenceNumber: seq },
  };
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd newfrontend && npm test -- reducer`
Expected: PASS (existing reducer tests + the 3 new ones).

- [ ] **Step 6: Build**

Run: `cd newfrontend && npm run build`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add newfrontend/src/types.ts newfrontend/src/stream/reducer.ts newfrontend/src/stream/__tests__/reducer.test.ts
git commit -m "feat(newfrontend): reducer tracks sequence cursor + handles response.incomplete (cancelled)"
```

---

## Task 2: SSE reader + responses API client

A minimal chunked-SSE parser and the resume/cancel HTTP clients.

**Files:**
- Create: `src/lib/sse.ts`, `src/api/responses.ts`
- Test: `src/lib/__tests__/sse.test.ts`, `src/api/__tests__/responses.test.ts`

**Interfaces:**
- Produces:
  - `parseSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<unknown>` — yields each `data:` line's parsed JSON, across chunk boundaries, skipping `[DONE]`.
  - `cancelResponse(id: string): Promise<void>` — `POST /v1/responses/{id}/cancel`; resolves on 2xx **and** 404 (already finished), throws on other failures.
  - `streamResume(id: string, startingAfter: number, signal: AbortSignal): AsyncIterable<unknown>` — `GET /v1/responses/{id}?stream=true&starting_after=N`, yields parsed events; throws on non-2xx (e.g. 409 not resumable).

- [ ] **Step 1: Write the failing tests**

`src/lib/__tests__/sse.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { parseSSE } from "../sse";

function streamFrom(parts: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(c) {
      for (const p of parts) c.enqueue(enc.encode(p));
      c.close();
    },
  });
}

async function collect(it: AsyncIterable<unknown>) {
  const out: unknown[] = [];
  for await (const x of it) out.push(x);
  return out;
}

describe("parseSSE", () => {
  it("yields each data event as parsed JSON", async () => {
    const out = await collect(
      parseSSE(streamFrom(['data: {"a":1}\n\n', 'data: {"b":2}\n\n']))
    );
    expect(out).toEqual([{ a: 1 }, { b: 2 }]);
  });

  it("reassembles events split across chunk boundaries", async () => {
    const out = await collect(parseSSE(streamFrom(['data: {"a":', '1}\n', "\n"])));
    expect(out).toEqual([{ a: 1 }]);
  });

  it("skips [DONE] sentinels", async () => {
    const out = await collect(
      parseSSE(streamFrom(['data: {"a":1}\n\n', "data: [DONE]\n\n"]))
    );
    expect(out).toEqual([{ a: 1 }]);
  });
});
```

`src/api/__tests__/responses.test.ts`:

```ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { cancelResponse, streamResume } from "../responses";

function streamFrom(parts: string[]): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(c) {
      for (const p of parts) c.enqueue(enc.encode(p));
      c.close();
    },
  });
}

describe("responses api", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("cancelResponse POSTs to the cancel path and tolerates 404", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 404 } as Response);
    vi.stubGlobal("fetch", fetchMock);
    await cancelResponse("resp_1");
    expect(String(fetchMock.mock.calls[0][0])).toContain("/v1/responses/resp_1/cancel");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "POST" });
  });

  it("cancelResponse throws on a real failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500 } as Response));
    await expect(cancelResponse("resp_1")).rejects.toThrow();
  });

  it("streamResume requests the cursor URL and yields parsed events", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      body: streamFrom(['data: {"type":"response.completed"}\n\n']),
    } as unknown as Response);
    vi.stubGlobal("fetch", fetchMock);
    const out: unknown[] = [];
    for await (const e of streamResume("resp_1", 7, new AbortController().signal)) out.push(e);
    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain("/v1/responses/resp_1");
    expect(url).toContain("stream=true");
    expect(url).toContain("starting_after=7");
    expect(out).toEqual([{ type: "response.completed" }]);
  });

  it("streamResume throws when the run is not resumable (409)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 409 } as Response));
    const it = streamResume("resp_x", 0, new AbortController().signal);
    await expect((async () => { for await (const _ of it) { /* drain */ } })()).rejects.toThrow();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd newfrontend && npm test -- "sse|responses"`
Expected: FAIL — cannot resolve `../sse` / `../responses`.

- [ ] **Step 3: Implement `src/lib/sse.ts`**

```ts
/** Parse a streaming SSE body into the JSON payloads of its `data:` lines,
 *  reassembling events that span read-chunk boundaries. */
export async function* parseSSE(
  stream: ReadableStream<Uint8Array>
): AsyncGenerator<unknown> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (payload && payload !== "[DONE]") {
            yield JSON.parse(payload);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
```

- [ ] **Step 4: Implement `src/api/responses.ts`**

```ts
import { parseSSE } from "../lib/sse";

export async function cancelResponse(id: string): Promise<void> {
  const res = await fetch(`/v1/responses/${encodeURIComponent(id)}/cancel`, {
    method: "POST",
  });
  // 404 == the run already finished/unknown server-side — treat as a no-op.
  if (!res.ok && res.status !== 404) {
    throw new Error(`cancel failed: ${res.status}`);
  }
}

export function streamResume(
  id: string,
  startingAfter: number,
  signal: AbortSignal
): AsyncIterable<unknown> {
  return (async function* () {
    const res = await fetch(
      `/v1/responses/${encodeURIComponent(id)}?stream=true&starting_after=${startingAfter}`,
      { signal }
    );
    if (!res.ok || !res.body) {
      throw new Error(`resume failed: ${res.status}`);
    }
    yield* parseSSE(res.body);
  })();
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd newfrontend && npm test -- "sse|responses"`
Expected: PASS (7 tests). If `ReadableStream`/`TextEncoder`/`TextDecoder` are missing under jsdom, they are Node 18+ globals (Node 24 here) and available in vitest; no polyfill should be needed.

- [ ] **Step 6: Build + full suite**

Run: `cd newfrontend && npm run build && npm test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add newfrontend/src/lib/sse.ts newfrontend/src/api/responses.ts newfrontend/src/lib/__tests__/sse.test.ts newfrontend/src/api/__tests__/responses.test.ts
git commit -m "feat(newfrontend): SSE reader + resume/cancel API clients"
```

---

## Task 3: `useResponsesChat` — background, server-side cancel, resume

Rewrite the hook: background send, one shared consume loop, Stop→cancel (deferred until the id is known), anchors-on-`cancelled`-or-`completed`, and `resumeIfInterrupted`.

**Files:**
- Rewrite: `src/hooks/useResponsesChat.ts`
- Test: `src/hooks/__tests__/useResponsesChat.test.tsx` (rewrite)

**Interfaces:**
- Consumes: `streamResponse` (`../api/client`), `cancelResponse`/`streamResume` (`../api/responses`), `getUserId`, `useChatStore`, `useConversationsStore`, `initialStreamState`/`reduceStreamEvent`/`StreamState`.
- Produces: `useResponsesChat()` → `{ send(text), stop(), regenerate(), isStreaming, resumeIfInterrupted() }`. `send` sets `background:true`; `stop()` calls `cancelResponse(responseId)` (or defers via a pending flag until `response.created` yields the id); the terminal advances anchors on `completed`/`cancelled` and refreshes the sidebar; `resumeIfInterrupted()` resumes a still-`streaming` last message from `lastSequenceNumber`.

- [ ] **Step 1: Write the failing test** — rewrite `src/hooks/__tests__/useResponsesChat.test.tsx`

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";

vi.mock("../../api/client", () => ({ streamResponse: vi.fn() }));
vi.mock("../../api/responses", () => ({ cancelResponse: vi.fn(), streamResume: vi.fn() }));
vi.mock("../../lib/user", () => ({ getUserId: () => "u1" }));
vi.mock("../../store/conversations", () => ({
  useConversationsStore: { getState: () => ({ refresh: vi.fn() }) },
}));

import { useResponsesChat } from "../useResponsesChat";
import { useChatStore } from "../../store/chat";
import * as client from "../../api/client";
import * as responsesApi from "../../api/responses";

function streamOf(events: any[]): AsyncIterable<any> {
  return {
    async *[Symbol.asyncIterator]() {
      for (const e of events) yield e;
    },
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  useChatStore.getState().reset();
});

describe("useResponsesChat (resilient)", () => {
  it("send runs in background and advances anchors on completion", async () => {
    (client.streamResponse as any).mockReturnValue(
      streamOf([
        { type: "response.created", response: { id: "resp_1", conversation: { id: "conv_1" } }, sequence_number: 1 },
        { type: "response.output_text.delta", delta: "hi", sequence_number: 2 },
        { type: "response.completed", response: { id: "resp_1", conversation: { id: "conv_1" }, status: "completed" }, sequence_number: 3 },
      ])
    );
    const { result } = renderHook(() => useResponsesChat());
    await act(async () => { await result.current.send("hello"); });
    const st = useChatStore.getState();
    expect(st.messages[1].text).toBe("hi");
    expect(st.messages[1].status).toBe("completed");
    expect(st.conversationId).toBe("conv_1");
    expect(st.lastResponseId).toBe("resp_1");
    const params = (client.streamResponse as any).mock.calls[0][0];
    expect(params.background).toBe(true);
    expect(params.user_id).toBe("u1");
  });

  it("stop cancels server-side once the response id is known", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => (release = r));
    (client.streamResponse as any).mockReturnValue({
      async *[Symbol.asyncIterator]() {
        yield { type: "response.created", response: { id: "resp_9", conversation: { id: "c" } }, sequence_number: 1 };
        yield { type: "response.output_text.delta", delta: "partial", sequence_number: 2 };
        await gate;
        yield { type: "response.incomplete", response: { id: "resp_9", conversation: { id: "c" }, status: "cancelled" }, sequence_number: 3 };
      },
    });
    const { result } = renderHook(() => useResponsesChat());
    let p: Promise<void>;
    await act(async () => { p = result.current.send("hello"); await Promise.resolve(); });
    act(() => result.current.stop());
    expect(responsesApi.cancelResponse).toHaveBeenCalledWith("resp_9");
    release();
    await act(async () => { await p; });
    const st = useChatStore.getState();
    expect(st.messages[1].status).toBe("cancelled");
    expect(st.messages[1].text).toBe("partial");
    // cancelled turn is continuable: anchors advanced
    expect(st.conversationId).toBe("c");
    expect(st.lastResponseId).toBe("resp_9");
  });

  it("resumeIfInterrupted resumes a streaming message from its cursor", async () => {
    // seed a half-streamed assistant message in the store
    useChatStore.setState({
      messages: [
        { id: "u", role: "user", text: "q", reasoning: "", reasoningStatus: "idle", status: "completed" },
        { id: "resp_5", role: "assistant", text: "par", reasoning: "", reasoningStatus: "idle", status: "streaming", responseId: "resp_5", lastSequenceNumber: 4 },
      ],
    });
    (responsesApi.streamResume as any).mockReturnValue(
      streamOf([
        { type: "response.output_text.delta", delta: "tial", sequence_number: 5 },
        { type: "response.completed", response: { id: "resp_5", conversation: { id: "c5" }, status: "completed" }, sequence_number: 6 },
      ])
    );
    const { result } = renderHook(() => useResponsesChat());
    await act(async () => { await result.current.resumeIfInterrupted(); });
    expect(responsesApi.streamResume).toHaveBeenCalledWith("resp_5", 4, expect.anything());
    const st = useChatStore.getState();
    expect(st.messages[1].text).toBe("partial");
    expect(st.messages[1].status).toBe("completed");
    expect(st.lastResponseId).toBe("resp_5");
  });

  it("resumeIfInterrupted is a no-op when the last message is not streaming", async () => {
    useChatStore.setState({
      messages: [{ id: "a", role: "assistant", text: "done", reasoning: "", reasoningStatus: "idle", status: "completed", responseId: "r" }],
    });
    const { result } = renderHook(() => useResponsesChat());
    await act(async () => { await result.current.resumeIfInterrupted(); });
    expect(responsesApi.streamResume).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd newfrontend && npm test -- useResponsesChat`
Expected: FAIL — `result.current.resumeIfInterrupted` is undefined / `background` not sent / Stop doesn't call `cancelResponse`.

- [ ] **Step 3: Rewrite `src/hooks/useResponsesChat.ts`**

```ts
import { useCallback, useRef, useState } from "react";
import { streamResponse } from "../api/client";
import { cancelResponse, streamResume } from "../api/responses";
import { getUserId } from "../lib/user";
import { useChatStore } from "../store/chat";
import { useConversationsStore } from "../store/conversations";
import {
  initialStreamState,
  reduceStreamEvent,
  type StreamState,
} from "../stream/reducer";
import type { ChatMessage } from "../types";

function tempId(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2)}`;
}

function patchFromState(s: StreamState): Partial<ChatMessage> {
  return {
    text: s.message.text,
    reasoning: s.message.reasoning,
    reasoningStatus: s.message.reasoningStatus,
    status: s.message.status,
    responseId: s.message.responseId,
    usage: s.message.usage,
    error: s.message.error,
    lastSequenceNumber: s.message.lastSequenceNumber,
  };
}

export function useResponsesChat() {
  const [isStreaming, setIsStreaming] = useState(false);
  const currentResponseId = useRef<string | undefined>(undefined);
  const cancelPending = useRef(false);
  const resuming = useRef(false);
  const lastInput = useRef("");

  // Shared loop: fold each event into the store's last message; fire a deferred
  // cancel the moment the response id is known.
  const consume = useCallback(
    async (stream: AsyncIterable<unknown>, seed: StreamState) => {
      let state = seed;
      for await (const event of stream) {
        state = reduceStreamEvent(state, event as never);
        if (state.responseId) {
          currentResponseId.current = state.responseId;
          if (cancelPending.current) {
            cancelPending.current = false;
            void cancelResponse(state.responseId);
          }
        }
        useChatStore.getState().updateLast(patchFromState(state));
      }
      return state;
    },
    []
  );

  const finalize = useCallback((state: StreamState) => {
    const s = state.message.status;
    // A cancelled turn is a real, persisted, continuable response — advance
    // anchors exactly as for completed.
    if (s === "completed" || s === "cancelled") {
      useChatStore.getState().setAnchors({
        conversationId: state.conversationId,
        lastResponseId: state.responseId,
      });
      void useConversationsStore.getState().refresh();
    }
  }, []);

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      const chat = useChatStore.getState();
      lastInput.current = trimmed;
      currentResponseId.current = undefined;
      cancelPending.current = false;

      const userMsg: ChatMessage = {
        id: tempId("user"),
        role: "user",
        text: trimmed,
        reasoning: "",
        reasoningStatus: "idle",
        status: "completed",
      };
      const assistantMsg: ChatMessage = {
        ...initialStreamState(tempId("assistant")).message,
      };
      chat.appendMessage(userMsg);
      chat.appendMessage(assistantMsg);
      chat.setStatus("streaming");
      setIsStreaming(true);

      const controller = new AbortController();
      let state = initialStreamState(assistantMsg.id);
      try {
        const stream = streamResponse(
          {
            model: chat.model,
            input: trimmed,
            user_id: getUserId(),
            conversation: chat.conversationId,
            previous_response_id: chat.lastResponseId,
            background: true,
          } as never,
          controller.signal
        );
        state = await consume(stream, state);
      } catch (err) {
        useChatStore.getState().updateLast({
          status: "failed",
          error: err instanceof Error ? err.message : "stream error",
        });
        setIsStreaming(false);
        useChatStore.getState().setStatus("idle");
        return;
      }
      setIsStreaming(false);
      useChatStore.getState().setStatus("idle");
      finalize(state);
    },
    [consume, finalize]
  );

  const stop = useCallback(() => {
    if (currentResponseId.current) {
      void cancelResponse(currentResponseId.current);
    } else {
      // response id not known yet — fire the cancel as soon as it arrives.
      cancelPending.current = true;
    }
  }, []);

  const regenerate = useCallback(async () => {
    if (lastInput.current) await send(lastInput.current);
  }, [send]);

  const resumeIfInterrupted = useCallback(async () => {
    if (resuming.current) return;
    const chat = useChatStore.getState();
    const last = chat.messages[chat.messages.length - 1];
    if (
      !last ||
      last.role !== "assistant" ||
      last.status !== "streaming" ||
      !last.responseId
    ) {
      return;
    }
    resuming.current = true;
    setIsStreaming(true);
    currentResponseId.current = last.responseId;
    const controller = new AbortController();
    const seed: StreamState = {
      message: last,
      responseId: last.responseId,
      conversationId: chat.conversationId,
      lastSequenceNumber: last.lastSequenceNumber ?? 0,
    };
    let state = seed;
    try {
      const stream = streamResume(
        last.responseId,
        seed.lastSequenceNumber,
        controller.signal
      );
      state = await consume(stream, state);
    } catch {
      // 409 (evicted/finished) or a network error: leave the bubble as-is.
      // The run finished server-side; a conversation reload shows the final.
      resuming.current = false;
      setIsStreaming(false);
      return;
    }
    resuming.current = false;
    setIsStreaming(false);
    useChatStore.getState().setStatus("idle");
    finalize(state);
  }, [consume, finalize]);

  return { send, stop, regenerate, isStreaming, resumeIfInterrupted };
}
```

NOTE: `streamResponse`'s typed `ResponseStreamParams` does not declare `background`; the `as never` cast on the params object lets the extra field through to the wire body (the backend's lenient `ResponsesRequest` accepts it). The local `AbortController` only tears down the *subscriber connection* — the run is server-owned, so Stop uses `cancelResponse`, not the abort.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd newfrontend && npm test -- useResponsesChat`
Expected: PASS (4 tests).

- [ ] **Step 5: Build + full suite**

Run: `cd newfrontend && npm run build && npm test`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add newfrontend/src/hooks/useResponsesChat.ts newfrontend/src/hooks/__tests__/useResponsesChat.test.tsx
git commit -m "feat(newfrontend): background runs + server-side cancel + resume in useResponsesChat"
```

---

## Task 4: Reconnect wiring + cancelled rendering

Wire `resumeIfInterrupted` to the browser's reconnect signals and render the cancelled state.

**Files:**
- Modify: `src/components/ChatView.tsx`, `src/components/AssistantMessage.tsx`
- Test: `src/components/__tests__/AssistantMessage.test.tsx` (new)

**Interfaces:**
- Consumes: `useResponsesChat().resumeIfInterrupted` (Task 3).
- Produces: ChatView calls `resumeIfInterrupted()` on mount, `visibilitychange`→visible, and `online`; `AssistantMessage` renders a "cancelled" note (and still shows `MessageControls`, since the turn is continuable).

- [ ] **Step 1: Write the failing test** — `src/components/__tests__/AssistantMessage.test.tsx`

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { AssistantMessage } from "../AssistantMessage";
import type { ChatMessage } from "../../types";

function msg(over: Partial<ChatMessage>): ChatMessage {
  return {
    id: "a",
    role: "assistant",
    text: "",
    reasoning: "",
    reasoningStatus: "idle",
    status: "completed",
    ...over,
  };
}

describe("AssistantMessage cancelled", () => {
  beforeEach(() => {
    Object.assign(navigator, { clipboard: { writeText: vi.fn().mockResolvedValue(undefined) } });
  });

  it("shows a cancelled note and the partial text", () => {
    render(<AssistantMessage message={msg({ status: "cancelled", text: "partial answer" })} />);
    expect(screen.getByText("partial answer")).toBeInTheDocument();
    expect(screen.getByText(/cancelled/i)).toBeInTheDocument();
  });

  it("still offers copy/regenerate for a cancelled (continuable) turn", () => {
    render(<AssistantMessage message={msg({ status: "cancelled", text: "x" })} onRegenerate={() => {}} />);
    expect(screen.getByRole("button", { name: /copy/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /regenerate/i })).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd newfrontend && npm test -- AssistantMessage`
Expected: FAIL — no "cancelled" note; `MessageControls` only renders for `completed`.

- [ ] **Step 3: Update `src/components/AssistantMessage.tsx`**

Replace the body below the `Markdown`/error block so cancelled is rendered and controls show for `completed` **or** `cancelled`:

```tsx
import type { ChatMessage } from "../types";
import { Markdown } from "./Markdown";
import { CollapsibleReasoning } from "./CollapsibleReasoning";
import { MessageControls } from "./MessageControls";

export function AssistantMessage({
  message,
  onRegenerate,
}: {
  message: ChatMessage;
  onRegenerate?: () => void;
}) {
  const showControls =
    message.status === "completed" || message.status === "cancelled";
  return (
    <div className="flex flex-col items-start">
      <div className="max-w-[80%]">
        <CollapsibleReasoning
          reasoning={message.reasoning}
          status={message.reasoningStatus}
        />
        {message.status === "failed" ? (
          <div className="rounded-md bg-red-50 px-3 py-2 text-red-700">
            {message.error || "Something went wrong."}
          </div>
        ) : (
          <Markdown content={message.text} />
        )}
        {message.status === "stopped" && (
          <div className="mt-1 text-xs italic text-gray-400">stopped</div>
        )}
        {message.status === "cancelled" && (
          <div className="mt-1 text-xs italic text-gray-400">cancelled</div>
        )}
        {showControls && (
          <MessageControls text={message.text} onRegenerate={onRegenerate} />
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Wire reconnect in `src/components/ChatView.tsx`**

```tsx
import { useEffect } from "react";
import { useChatStore } from "../store/chat";
import { useResponsesChat } from "../hooks/useResponsesChat";
import { MessageList } from "./MessageList";
import { Composer } from "./Composer";
import { ModelSelector } from "./ModelSelector";

export function ChatView() {
  const model = useChatStore((s) => s.model);
  const setModel = useChatStore((s) => s.setModel);
  const { send, stop, regenerate, isStreaming, resumeIfInterrupted } =
    useResponsesChat();

  // Resume an interrupted in-flight answer when the tab/network comes back.
  useEffect(() => {
    const tryResume = () => {
      if (document.visibilityState === "visible") void resumeIfInterrupted();
    };
    tryResume(); // also on mount (no-op unless a streaming message exists)
    document.addEventListener("visibilitychange", tryResume);
    window.addEventListener("online", tryResume);
    return () => {
      document.removeEventListener("visibilitychange", tryResume);
      window.removeEventListener("online", tryResume);
    };
  }, [resumeIfInterrupted]);

  return (
    <div className="flex h-full flex-1 flex-col">
      <div className="flex items-center justify-between border-b border-gray-200 p-3">
        <span className="font-medium">Agent Chat</span>
        <ModelSelector model={model} onChange={setModel} />
      </div>
      <MessageList onRegenerate={regenerate} />
      <Composer onSend={send} onStop={stop} isStreaming={isStreaming} />
    </div>
  );
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd newfrontend && npm test -- AssistantMessage`
Expected: PASS (2 tests).

- [ ] **Step 6: Run the full suite + build**

Run: `cd newfrontend && npm test && npm run build`
Expected: PASS (all tests; existing `App.test.tsx` still green — `ChatView` renders the same elements).

- [ ] **Step 7: Commit**

```bash
git add newfrontend/src/components/ChatView.tsx newfrontend/src/components/AssistantMessage.tsx newfrontend/src/components/__tests__/AssistantMessage.test.tsx
git commit -m "feat(newfrontend): reconnect on visibility/online + render cancelled turns"
```

---

## Self-Review (completed against the spec)

- **Reducer tracks `lastSequenceNumber` (monotonic, every event) + `response.incomplete`→`cancelled`; `MessageStatus` += cancelled; `ChatMessage.lastSequenceNumber`** → Task 1.
- **SSE reader + `cancelResponse` + `streamResume(starting_after)`** → Task 2 (404-tolerant cancel; 409 throws on resume).
- **Background send; one consume loop; Stop→server cancel (deferred until id known); anchors advance on completed OR cancelled; `resumeIfInterrupted` from the message cursor** → Task 3.
- **Reconnect on visibility/online/mount; cancelled rendered (with controls, since continuable)** → Task 4.
- **Scope honored:** same-session resume only (no hard-reload re-attach, no localStorage in-flight pointer); no data loss relies on the backend's detached-run persistence (prerequisite plan). Regenerate keeps its v1 behavior (documented limitation, unchanged).

Type/signature consistency across tasks: `StreamState.lastSequenceNumber`, `parseSSE(stream)`, `cancelResponse(id)`, `streamResume(id, startingAfter, signal)`, `patchFromState`, hook return `{ send, stop, regenerate, isStreaming, resumeIfInterrupted }` all match between definition and use.
