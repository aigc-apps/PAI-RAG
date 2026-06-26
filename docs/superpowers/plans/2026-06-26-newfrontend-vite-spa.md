# newfrontend Vite SPA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a brand-new standalone agent-chat frontend at `newfrontend/` — a Vite + React + TypeScript SPA (no Next.js, no assistant-ui) that streams the lean service's `/v1/responses` (OpenAI Responses + `reasoning.summary`) and powers a conversation sidebar via the new Conversations API.

**Architecture:** A small SPA with a single testable core — a PURE stream reducer (`stream/reducer.ts`) that folds `response.*` SSE events into the in-flight assistant message. The `useResponsesChat` hook owns the official `openai` JS SDK stream (`client.responses.create({stream:true}, {signal})`), feeds events to the reducer, and exposes `send`/`stop`/`regenerate`. Two zustand stores hold chat state and the conversation list. Components render Markdown answers + a collapsible streaming reasoning panel. Continuation is anchored on `conversation_id` (+ `last_response_id` as a secondary hint), with a per-browser `user_id` (localStorage UUID) for sidebar isolation.

**Tech Stack:** Vite 6, React 19, TypeScript 5, Tailwind v4 (`@tailwindcss/vite`), Radix (`@radix-ui/react-collapsible`), `react-markdown` + `remark-gfm` + `react-syntax-highlighter`, `lucide-react`, `sonner`, `zustand`, the `openai` JS SDK (v5). Tests: Vitest 3 + `@testing-library/react` + `jsdom`. All commands run from `newfrontend/`.

**Reference spec:** `docs/superpowers/specs/2026-06-26-new-agent-chat-frontend-design.md` ("Frontend", "Data flow", "Stop semantics", "Testing"). **Prerequisite plan:** `2026-06-26-backend-conversations-api.md` must be implemented first (this SPA consumes `GET/DELETE /v1/conversations` and the `user_id` field). Branch: `personal/yfei/agent-core`.

## Global Constraints

- **Greenfield, isolated:** create everything under `newfrontend/`. Do NOT touch `frontend/` (the old Next.js app) or `backend/`.
- **No assistant-ui, no Next.js.** Pure Vite SPA.
- **The reducer (`stream/reducer.ts`) is pure** — `(state, event) => state`, no I/O, no SDK calls, no clock/random. It is the conformance gate against the serializer's event names.
- **Event names are the contract** the backend emits (verified from `backend/api/protocol/responses_serializer.py`): `response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`, `response.output_text.done`, `response.content_part.done`, `response.output_item.done`, `response.reasoning_summary_part.added`, `response.reasoning_summary_text.delta`, `response.reasoning_summary_text.done`, `response.reasoning_summary_part.done`, `response.function_call_arguments.delta`, `response.function_call_arguments.done`, `response.completed`, `response.failed`. Reasoning streams via the **`reasoning_summary_*`** family (NOT `reasoning_text`). Reasoning items are surfaced; `function_call*` events are IGNORED in v1.
- **Continuation anchor = `conversation_id`** (kept client-side, sent on every follow-up) with `previous_response_id` as a secondary hint. **Stop is local-only:** aborting keeps a client-only "stopped" bubble and does NOT advance `conversation_id`/`last_response_id`.
- **Per-browser `user_id`:** a localStorage UUID, sent on `/v1/responses` and as `?user_id=` on `GET /v1/conversations`. Not a security boundary.
- **TypeScript strict mode on.** `npm run build` (tsc + vite build) and `npm test` (vitest run) must both pass at the end of every task.
- Default model id: `gpt-4o-mini` (matches the backend `Settings.default_model`); the selector also offers `gpt-4o`.

---

## Verified backend stream shapes (do not re-derive)

The lean serializer emits these field shapes (the SDK parses them into typed events; the reducer reads these fields):

- `response.created` / `response.completed` / `response.failed` carry `event.response` = a `Response` with `id: string`, `conversation: { id } | null`, `status`, `usage: { input_tokens, output_tokens, total_tokens } | null`, `error: { code, message } | null`.
- `response.output_text.delta` → `{ delta: string, item_id, output_index, content_index }`.
- `response.reasoning_summary_text.delta` → `{ delta: string, item_id, output_index, summary_index }`.
- `response.reasoning_summary_part.done` → reasoning summary finished (use to flip reasoning to "done").
- `response.output_item.done` with `item.type === "reasoning"` also marks reasoning done; with `item.type === "message"` marks the answer item done.

Continuation request body for `/v1/responses`: `{ model, input: <text>, user_id, store: true, conversation?: <id>, previous_response_id?: <id>, stream: true }`. First turn of a fresh chat sends neither `conversation` nor `previous_response_id`.

Conversations API (from the prerequisite plan):
- `GET /v1/conversations?user_id=&limit=&offset=` → `{ data: [{ id, title, created_at, updated_at, last_response_id }] }`.
- `GET /v1/conversations/{id}` → `{ id, title, created_at, updated_at, latest_response_id, messages: [ {role:"user", text, response_id} | {role:"assistant", text, reasoning, response_id, previous_response_id, status} ] }`.
- `DELETE /v1/conversations/{id}` → `{ id, object:"conversation.deleted", deleted:true }` / 404.

---

## File Structure

```text
newfrontend/
  package.json
  tsconfig.json
  tsconfig.node.json
  vite.config.ts            # @vitejs/plugin-react + @tailwindcss/vite; dev proxy /v1 -> :8000; vitest config
  vitest.setup.ts           # @testing-library/jest-dom
  index.html
  src/
    main.tsx                # React root
    index.css               # @import "tailwindcss"
    lib/
      user.ts               # stable localStorage user_id
      cn.ts                 # className join helper
    types.ts                # shared UI types (ChatMessage, ConversationSummary, ...)
    api/
      client.ts             # openai SDK client + streamResponse(params, signal)
      conversations.ts      # GET list / GET one / DELETE
    stream/
      reducer.ts            # PURE (state, response.* event) -> stream state
    store/
      chat.ts               # zustand: messages, status, model, conversationId, lastResponseId
      conversations.ts      # zustand: list, selectedId, load/select/delete
    hooks/
      useResponsesChat.ts   # send a turn, consume the stream, stop/regenerate
    components/
      App.tsx
      Sidebar.tsx
      ChatView.tsx
      MessageList.tsx
      UserMessage.tsx
      AssistantMessage.tsx
      CollapsibleReasoning.tsx
      Markdown.tsx
      Composer.tsx
      ModelSelector.tsx
      MessageControls.tsx
  src/**/__tests__/*.test.ts(x)
```

---

## Task 1: Scaffold the Vite + React + TS app

A buildable, testable skeleton: tooling, Tailwind, a mounting root, and a passing trivial test.

**Files:**
- Create: `newfrontend/package.json`, `tsconfig.json`, `tsconfig.node.json`, `vite.config.ts`, `vitest.setup.ts`, `index.html`, `src/main.tsx`, `src/index.css`, `src/lib/cn.ts`
- Test: `newfrontend/src/lib/__tests__/cn.test.ts`

**Interfaces:**
- Produces: `cn(...classes: (string | false | null | undefined)[]) => string` (used by all components); a Vite dev server proxying `/v1` to `http://localhost:8000`; `npm run build` and `npm test` scripts.

- [ ] **Step 1: Create `newfrontend/package.json`**

```json
{
  "name": "newfrontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "@radix-ui/react-collapsible": "^1.1.10",
    "lucide-react": "^0.503.0",
    "openai": "^5.0.0",
    "react": "^19.1.2",
    "react-dom": "^19.1.2",
    "react-markdown": "^10.1.0",
    "react-syntax-highlighter": "^16.1.0",
    "remark-gfm": "^4.0.1",
    "sonner": "^2.0.7",
    "zustand": "^5.0.5"
  },
  "devDependencies": {
    "@tailwindcss/vite": "^4.0.0",
    "@testing-library/jest-dom": "^6.4.0",
    "@testing-library/react": "^16.0.0",
    "@testing-library/user-event": "^14.5.0",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@types/react-syntax-highlighter": "^15.5.13",
    "@vitejs/plugin-react": "^4.3.0",
    "jsdom": "^25.0.0",
    "tailwindcss": "^4.0.0",
    "typescript": "^5.6.0",
    "vite": "^6.0.0",
    "vitest": "^3.0.0"
  }
}
```

- [ ] **Step 2: Create `newfrontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": false,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "types": ["vitest/globals", "@testing-library/jest-dom"]
  },
  "include": ["src", "vitest.setup.ts"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

- [ ] **Step 3: Create `newfrontend/tsconfig.node.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2023"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "noEmit": true
  },
  "include": ["vite.config.ts"]
}
```

- [ ] **Step 4: Create `newfrontend/vite.config.ts`**

```ts
/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/v1": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
  },
});
```

- [ ] **Step 5: Create `newfrontend/vitest.setup.ts`**

```ts
import "@testing-library/jest-dom/vitest";
```

- [ ] **Step 6: Create `newfrontend/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Agent Chat</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 7: Create `newfrontend/src/index.css`**

```css
@import "tailwindcss";

html,
body,
#root {
  height: 100%;
  margin: 0;
}
```

- [ ] **Step 8: Create `newfrontend/src/lib/cn.ts`**

```ts
export function cn(
  ...classes: (string | false | null | undefined)[]
): string {
  return classes.filter(Boolean).join(" ");
}
```

- [ ] **Step 9: Create `newfrontend/src/main.tsx`** (minimal placeholder root — the real App lands in Task 7)

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";

function Placeholder() {
  return <div>Agent Chat</div>;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Placeholder />
  </StrictMode>
);
```

- [ ] **Step 10: Write the failing test** — `newfrontend/src/lib/__tests__/cn.test.ts`

```ts
import { describe, it, expect } from "vitest";
import { cn } from "../cn";

describe("cn", () => {
  it("joins truthy class names and drops falsy", () => {
    expect(cn("a", false, "b", null, undefined, "c")).toBe("a b c");
  });
});
```

- [ ] **Step 11: Install deps and run the test**

Run: `cd newfrontend && npm install && npm test`
Expected: PASS (1 test). If `npm install` is blocked offline, note it and stop — the rest of the plan needs the toolchain.

- [ ] **Step 12: Verify the build**

Run: `cd newfrontend && npm run build`
Expected: tsc + vite build succeed (a `dist/` is produced).

- [ ] **Step 13: Commit**

```bash
git add newfrontend/package.json newfrontend/package-lock.json newfrontend/tsconfig.json newfrontend/tsconfig.node.json newfrontend/vite.config.ts newfrontend/vitest.setup.ts newfrontend/index.html newfrontend/src/index.css newfrontend/src/main.tsx newfrontend/src/lib/cn.ts newfrontend/src/lib/__tests__/cn.test.ts
git commit -m "feat(newfrontend): scaffold Vite + React + TS SPA with Tailwind v4 + Vitest"
```

---

## Task 2: Shared types + API clients

The data layer: shared UI types, the openai SDK client (stream factory + abort), and the Conversations REST client, with a stable per-browser `user_id`.

**Files:**
- Create: `newfrontend/src/types.ts`, `newfrontend/src/lib/user.ts`, `newfrontend/src/api/client.ts`, `newfrontend/src/api/conversations.ts`
- Test: `newfrontend/src/api/__tests__/conversations.test.ts`, `newfrontend/src/lib/__tests__/user.test.ts`

**Interfaces:**
- Produces:
  - `types.ts`: `MessageStatus = "streaming" | "completed" | "failed" | "stopped"`; `ReasoningStatus = "idle" | "streaming" | "done"`; `ChatMessage` (see below); `ConversationSummary = { id; title: string | null; created_at: string | null; updated_at: string | null; last_response_id: string | null }`; `ConversationDetail = { id; title; created_at; updated_at; latest_response_id: string | null; messages: ChatMessage[] }`.
  - `lib/user.ts`: `getUserId(): string` (stable localStorage UUID).
  - `api/client.ts`: `streamResponse(params: ResponseStreamParams, signal: AbortSignal): AsyncIterable<ResponseStreamEvent>` where `ResponseStreamParams = { model: string; input: string; user_id: string; conversation?: string; previous_response_id?: string }`.
  - `api/conversations.ts`: `listConversations(userId: string): Promise<ConversationSummary[]>`; `getConversation(id: string): Promise<ConversationDetail>`; `deleteConversation(id: string): Promise<void>`.

- [ ] **Step 1: Create `newfrontend/src/types.ts`**

```ts
export type MessageStatus = "streaming" | "completed" | "failed" | "stopped";
export type ReasoningStatus = "idle" | "streaming" | "done";

export interface ChatMessage {
  /** local list key; equals responseId for assistant turns once known */
  id: string;
  role: "user" | "assistant";
  text: string;
  reasoning: string;
  reasoningStatus: ReasoningStatus;
  status: MessageStatus;
  responseId?: string;
  previousResponseId?: string;
  error?: string;
  usage?: { input: number; output: number; total: number };
}

export interface ConversationSummary {
  id: string;
  title: string | null;
  created_at: string | null;
  updated_at: string | null;
  last_response_id: string | null;
}

export interface ConversationDetail {
  id: string;
  title: string | null;
  created_at: string | null;
  updated_at: string | null;
  latest_response_id: string | null;
  messages: ChatMessage[];
}
```

- [ ] **Step 2: Write the failing test for `user.ts`** — `newfrontend/src/lib/__tests__/user.test.ts`

```ts
import { describe, it, expect, beforeEach } from "vitest";
import { getUserId } from "../user";

describe("getUserId", () => {
  beforeEach(() => localStorage.clear());

  it("generates and persists a stable id", () => {
    const a = getUserId();
    const b = getUserId();
    expect(a).toBe(b);
    expect(a.length).toBeGreaterThan(0);
    expect(localStorage.getItem("agent-chat:user_id")).toBe(a);
  });
});
```

- [ ] **Step 3: Run to verify failure**

Run: `cd newfrontend && npm test -- user`
Expected: FAIL — cannot resolve `../user`.

- [ ] **Step 4: Create `newfrontend/src/lib/user.ts`**

```ts
const KEY = "agent-chat:user_id";

export function getUserId(): string {
  let id = localStorage.getItem(KEY);
  if (!id) {
    id =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `u_${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
    localStorage.setItem(KEY, id);
  }
  return id;
}
```

- [ ] **Step 5: Create `newfrontend/src/api/client.ts`**

```ts
import OpenAI from "openai";
import type { ResponseStreamEvent } from "openai/resources/responses/responses";

// The real API key lives server-side in the lean service; the browser only ever
// talks to our own service via the Vite dev proxy, so a placeholder + browser
// usage is acceptable for v1 (no auth layer yet).
const client = new OpenAI({
  baseURL: `${window.location.origin}/v1`,
  apiKey: "sk-noauth",
  dangerouslyAllowBrowser: true,
});

export interface ResponseStreamParams {
  model: string;
  input: string;
  user_id: string;
  conversation?: string;
  previous_response_id?: string;
}

export function streamResponse(
  params: ResponseStreamParams,
  signal: AbortSignal
): AsyncIterable<ResponseStreamEvent> {
  // store=true so the backend persists the turn + conversation row.
  return client.responses.create(
    { ...params, store: true, stream: true } as never,
    { signal }
  );
}
```

NOTE: the `as never` cast on the params keeps us from fighting the SDK's strict `ResponseCreateParams` union over our extra `user_id` field; the wire body is exactly what the backend's lenient `ResponsesRequest` accepts. `client.responses.create({stream:true}, {signal})` returns an async-iterable stream that aborts when `signal` fires.

- [ ] **Step 6: Write the failing test for conversations** — `newfrontend/src/api/__tests__/conversations.test.ts`

```ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  listConversations,
  getConversation,
  deleteConversation,
} from "../conversations";

function mockFetch(json: unknown, ok = true, status = 200) {
  return vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => json,
  } as Response);
}

describe("conversations api", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("lists conversations for a user", async () => {
    const fetchMock = mockFetch({
      data: [
        {
          id: "c1",
          title: "hi",
          created_at: "t",
          updated_at: "t",
          last_response_id: "r1",
        },
      ],
    });
    vi.stubGlobal("fetch", fetchMock);
    const out = await listConversations("u1");
    expect(out).toHaveLength(1);
    expect(out[0].id).toBe("c1");
    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain("/v1/conversations");
    expect(url).toContain("user_id=u1");
  });

  it("gets a conversation detail", async () => {
    vi.stubGlobal(
      "fetch",
      mockFetch({
        id: "c1",
        title: "hi",
        created_at: "t",
        updated_at: "t",
        latest_response_id: "r2",
        messages: [],
      })
    );
    const out = await getConversation("c1");
    expect(out.latest_response_id).toBe("r2");
  });

  it("throws on a 404 detail", async () => {
    vi.stubGlobal("fetch", mockFetch({}, false, 404));
    await expect(getConversation("nope")).rejects.toThrow();
  });

  it("deletes a conversation", async () => {
    const fetchMock = mockFetch({ deleted: true });
    vi.stubGlobal("fetch", fetchMock);
    await deleteConversation("c1");
    expect(String(fetchMock.mock.calls[0][0])).toContain("/v1/conversations/c1");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "DELETE" });
  });
});
```

- [ ] **Step 7: Run to verify failure**

Run: `cd newfrontend && npm test -- conversations`
Expected: FAIL — cannot resolve `../conversations`.

- [ ] **Step 8: Create `newfrontend/src/api/conversations.ts`**

```ts
import type { ConversationDetail, ConversationSummary } from "../types";

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    throw new Error(`request failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function listConversations(
  userId: string
): Promise<ConversationSummary[]> {
  const res = await fetch(
    `/v1/conversations?user_id=${encodeURIComponent(userId)}`
  );
  const body = await jsonOrThrow<{ data: ConversationSummary[] }>(res);
  return body.data;
}

export async function getConversation(
  id: string
): Promise<ConversationDetail> {
  const res = await fetch(`/v1/conversations/${encodeURIComponent(id)}`);
  return jsonOrThrow<ConversationDetail>(res);
}

export async function deleteConversation(id: string): Promise<void> {
  const res = await fetch(`/v1/conversations/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(`delete failed: ${res.status}`);
  }
}
```

NOTE: the detail endpoint returns assistant messages already shaped close to `ChatMessage` (`role`, `text`, `reasoning`, `response_id`, `previous_response_id`, `status`) and user messages as `{role, text, response_id}`. The store (Task 4) normalizes these wire rows into full `ChatMessage` objects (filling `id`/`reasoningStatus`/camelCase ids) when loading history — `getConversation` returns them as-is.

- [ ] **Step 9: Run to verify pass**

Run: `cd newfrontend && npm test -- conversations user`
Expected: PASS.

- [ ] **Step 10: Build**

Run: `cd newfrontend && npm run build`
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add newfrontend/src/types.ts newfrontend/src/lib/user.ts newfrontend/src/api newfrontend/src/lib/__tests__/user.test.ts
git commit -m "feat(newfrontend): shared types + openai stream client + conversations REST client"
```

---

## Task 3: The pure stream reducer (the testable core)

`stream/reducer.ts` folds each `response.*` event onto the in-flight assistant message. This is the most important unit in the SPA; test it exhaustively against recorded event sequences matching the serializer.

**Files:**
- Create: `newfrontend/src/stream/reducer.ts`
- Test: `newfrontend/src/stream/__tests__/reducer.test.ts`

**Interfaces:**
- Consumes: `ChatMessage`, `MessageStatus`, `ReasoningStatus` from `../types`.
- Produces:
  - `StreamState = { message: ChatMessage; conversationId?: string; responseId?: string }`.
  - `initialStreamState(id: string): StreamState` — a fresh assistant message (`status: "streaming"`, `reasoningStatus: "idle"`).
  - `reduceStreamEvent(state: StreamState, event: ResponseStreamEvent): StreamState` — pure; returns a new state. Unknown / ignored events (including all `function_call*`) return the state unchanged.

- [ ] **Step 1: Write the failing test** — `newfrontend/src/stream/__tests__/reducer.test.ts`

```ts
import { describe, it, expect } from "vitest";
import { initialStreamState, reduceStreamEvent } from "../reducer";

// Minimal recorded event sequences mirroring backend/api/protocol/responses_serializer.py.
// Typed loosely as `any` because we only feed the fields the reducer reads.
function fold(events: any[]) {
  let state = initialStreamState("tmp");
  for (const ev of events) state = reduceStreamEvent(state, ev as any);
  return state;
}

const created = (id: string, convId: string | null) => ({
  type: "response.created",
  response: { id, conversation: convId ? { id: convId } : null, status: "in_progress" },
});
const textDelta = (delta: string) => ({ type: "response.output_text.delta", delta });
const reasoningDelta = (delta: string) => ({
  type: "response.reasoning_summary_text.delta",
  delta,
});
const reasoningPartDone = () => ({ type: "response.reasoning_summary_part.done" });
const completed = (id: string, convId: string) => ({
  type: "response.completed",
  response: {
    id,
    conversation: { id: convId },
    status: "completed",
    usage: { input_tokens: 3, output_tokens: 5, total_tokens: 8 },
  },
});
const failed = (id: string, message: string) => ({
  type: "response.failed",
  response: {
    id,
    conversation: null,
    status: "failed",
    error: { code: "server_error", message },
  },
});

describe("reduceStreamEvent", () => {
  it("plain text turn captures ids, text, usage and completed status", () => {
    const s = fold([
      created("resp_1", "conv_1"),
      textDelta("Hell"),
      textDelta("o"),
      completed("resp_1", "conv_1"),
    ]);
    expect(s.responseId).toBe("resp_1");
    expect(s.conversationId).toBe("conv_1");
    expect(s.message.text).toBe("Hello");
    expect(s.message.responseId).toBe("resp_1");
    expect(s.message.status).toBe("completed");
    expect(s.message.usage).toEqual({ input: 3, output: 5, total: 8 });
    expect(s.message.reasoning).toBe("");
    expect(s.message.reasoningStatus).toBe("idle");
  });

  it("reasoning+text turn accumulates reasoning then flips it to done", () => {
    const s = fold([
      created("resp_2", "conv_2"),
      reasoningDelta("think "),
      reasoningDelta("hard"),
      reasoningPartDone(),
      textDelta("answer"),
      completed("resp_2", "conv_2"),
    ]);
    expect(s.message.reasoning).toBe("think hard");
    expect(s.message.reasoningStatus).toBe("done");
    expect(s.message.text).toBe("answer");
    expect(s.message.status).toBe("completed");
  });

  it("reasoning streaming status is set while deltas arrive", () => {
    const s = fold([created("resp_3", "c"), reasoningDelta("x")]);
    expect(s.message.reasoningStatus).toBe("streaming");
    expect(s.message.status).toBe("streaming");
  });

  it("output_item.done for a reasoning item also marks reasoning done", () => {
    const s = fold([
      created("resp_4", "c"),
      reasoningDelta("x"),
      { type: "response.output_item.done", item: { type: "reasoning" } },
    ]);
    expect(s.message.reasoningStatus).toBe("done");
  });

  it("failed turn sets failed status + error message", () => {
    const s = fold([created("resp_5", "c"), failed("resp_5", "kaboom")]);
    expect(s.message.status).toBe("failed");
    expect(s.message.error).toContain("kaboom");
  });

  it("ignores function_call events (tool UI out of scope)", () => {
    const s = fold([
      created("resp_6", "c"),
      { type: "response.function_call_arguments.delta", delta: '{"a":1}' },
      { type: "response.function_call_arguments.done", arguments: '{"a":1}', name: "get" },
      textDelta("done"),
      completed("resp_6", "c"),
    ]);
    expect(s.message.text).toBe("done");
  });

  it("is pure: does not mutate the input state", () => {
    const s0 = initialStreamState("tmp");
    const s1 = reduceStreamEvent(s0, textDelta("a") as any);
    expect(s0.message.text).toBe("");
    expect(s1.message.text).toBe("a");
    expect(s1).not.toBe(s0);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd newfrontend && npm test -- reducer`
Expected: FAIL — cannot resolve `../reducer`.

- [ ] **Step 3: Implement `newfrontend/src/stream/reducer.ts`**

```ts
import type { ResponseStreamEvent } from "openai/resources/responses/responses";
import type { ChatMessage } from "../types";

export interface StreamState {
  message: ChatMessage;
  conversationId?: string;
  responseId?: string;
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
    },
  };
}

// Narrow helper: read a field off a loosely-typed event without fighting the
// SDK's giant discriminated union in every branch.
function f(event: ResponseStreamEvent): Record<string, unknown> {
  return event as unknown as Record<string, unknown>;
}

export function reduceStreamEvent(
  state: StreamState,
  event: ResponseStreamEvent
): StreamState {
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

    case "response.completed": {
      const response = e.response as
        | {
            id?: string;
            conversation?: { id?: string } | null;
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
      return {
        ...state,
        responseId: response?.id ?? state.responseId,
        conversationId: response?.conversation?.id ?? state.conversationId,
        message: {
          ...msg,
          status: "completed",
          reasoningStatus: msg.reasoningStatus === "streaming" ? "done" : msg.reasoningStatus,
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
```

- [ ] **Step 4: Run to verify pass**

Run: `cd newfrontend && npm test -- reducer`
Expected: PASS (7 tests).

- [ ] **Step 5: Build**

Run: `cd newfrontend && npm run build`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add newfrontend/src/stream
git commit -m "feat(newfrontend): pure stream reducer folding response.* events into a message"
```

---

## Task 4: Zustand stores (chat + conversations)

App state: the chat store (messages, model, anchors) and the conversations store (list, selection). Plain state + actions, fully unit-tested. No network calls here except via the injected conversations API.

**Files:**
- Create: `newfrontend/src/store/chat.ts`, `newfrontend/src/store/conversations.ts`
- Test: `newfrontend/src/store/__tests__/chat.test.ts`, `newfrontend/src/store/__tests__/conversations.test.ts`

**Interfaces:**
- Consumes: `ChatMessage`, `ConversationSummary`, `ConversationDetail` (`../types`); `listConversations`/`getConversation`/`deleteConversation` (`../api/conversations`); `getUserId` (`../lib/user`).
- Produces:
  - `useChatStore` with state `{ messages: ChatMessage[]; status: "idle" | "streaming"; model: string; conversationId?: string; lastResponseId?: string }` and actions: `appendMessage(m)`, `updateLast(patch)` (patch the last assistant message), `setModel(m)`, `setAnchors({conversationId?, lastResponseId?})`, `reset()`, `loadHistory(detail)` (replace messages + set anchors from a `ConversationDetail`).
  - `useConversationsStore` with state `{ items: ConversationSummary[]; selectedId?: string }` and actions: `refresh()`, `select(id)`, `clearSelection()`, `remove(id)`.
  - `normalizeHistoryMessages(detail: ConversationDetail): ChatMessage[]` — exported pure helper that turns wire rows into full `ChatMessage`s.

- [ ] **Step 1: Write the failing test for the chat store** — `newfrontend/src/store/__tests__/chat.test.ts`

```ts
import { describe, it, expect, beforeEach } from "vitest";
import { useChatStore, normalizeHistoryMessages } from "../chat";
import type { ConversationDetail } from "../../types";

beforeEach(() => useChatStore.getState().reset());

describe("chat store", () => {
  it("appends and patches the last message", () => {
    const s = useChatStore.getState();
    s.appendMessage({
      id: "u1",
      role: "user",
      text: "hi",
      reasoning: "",
      reasoningStatus: "idle",
      status: "completed",
    });
    s.appendMessage({
      id: "a1",
      role: "assistant",
      text: "",
      reasoning: "",
      reasoningStatus: "idle",
      status: "streaming",
    });
    useChatStore.getState().updateLast({ text: "hello", status: "completed" });
    const msgs = useChatStore.getState().messages;
    expect(msgs).toHaveLength(2);
    expect(msgs[1].text).toBe("hello");
    expect(msgs[1].status).toBe("completed");
  });

  it("sets anchors and model", () => {
    useChatStore.getState().setAnchors({ conversationId: "c1", lastResponseId: "r1" });
    useChatStore.getState().setModel("gpt-4o");
    const st = useChatStore.getState();
    expect(st.conversationId).toBe("c1");
    expect(st.lastResponseId).toBe("r1");
    expect(st.model).toBe("gpt-4o");
  });

  it("loadHistory replaces messages and sets anchors", () => {
    const detail: ConversationDetail = {
      id: "c9",
      title: "t",
      created_at: null,
      updated_at: null,
      latest_response_id: "r9",
      messages: [
        { role: "user", text: "q", response_id: "r9" } as never,
        {
          role: "assistant",
          text: "a",
          reasoning: "why",
          response_id: "r9",
          previous_response_id: null,
          status: "completed",
        } as never,
      ],
    };
    useChatStore.getState().loadHistory(detail);
    const st = useChatStore.getState();
    expect(st.conversationId).toBe("c9");
    expect(st.lastResponseId).toBe("r9");
    expect(st.messages).toHaveLength(2);
    expect(st.messages[1].reasoning).toBe("why");
    expect(st.messages[1].reasoningStatus).toBe("done");
  });
});

describe("normalizeHistoryMessages", () => {
  it("fills ids, camelCases, and sets reasoningStatus", () => {
    const out = normalizeHistoryMessages({
      id: "c",
      title: null,
      created_at: null,
      updated_at: null,
      latest_response_id: "r1",
      messages: [
        {
          role: "assistant",
          text: "a",
          reasoning: "",
          response_id: "r1",
          previous_response_id: "r0",
          status: "completed",
        } as never,
      ],
    });
    expect(out[0].id).toBe("r1");
    expect(out[0].responseId).toBe("r1");
    expect(out[0].previousResponseId).toBe("r0");
    expect(out[0].reasoningStatus).toBe("idle"); // empty reasoning -> idle
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd newfrontend && npm test -- store/__tests__/chat`
Expected: FAIL — cannot resolve `../chat`.

- [ ] **Step 3: Implement `newfrontend/src/store/chat.ts`**

```ts
import { create } from "zustand";
import type { ChatMessage, ConversationDetail } from "../types";

interface WireHistoryMessage {
  role: "user" | "assistant";
  text: string;
  reasoning?: string;
  response_id: string;
  previous_response_id?: string | null;
  status?: ChatMessage["status"];
}

export function normalizeHistoryMessages(
  detail: ConversationDetail
): ChatMessage[] {
  const rows = detail.messages as unknown as WireHistoryMessage[];
  return rows.map((r, i) => {
    const reasoning = r.reasoning ?? "";
    return {
      id: r.role === "assistant" ? r.response_id : `${r.response_id}:u:${i}`,
      role: r.role,
      text: r.text ?? "",
      reasoning,
      reasoningStatus: reasoning ? "done" : "idle",
      status: r.status ?? "completed",
      responseId: r.response_id,
      previousResponseId: r.previous_response_id ?? undefined,
    };
  });
}

interface ChatState {
  messages: ChatMessage[];
  status: "idle" | "streaming";
  model: string;
  conversationId?: string;
  lastResponseId?: string;
  appendMessage: (m: ChatMessage) => void;
  updateLast: (patch: Partial<ChatMessage>) => void;
  setStatus: (status: "idle" | "streaming") => void;
  setModel: (model: string) => void;
  setAnchors: (a: { conversationId?: string; lastResponseId?: string }) => void;
  loadHistory: (detail: ConversationDetail) => void;
  reset: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  status: "idle",
  model: "gpt-4o-mini",
  conversationId: undefined,
  lastResponseId: undefined,

  appendMessage: (m) => set((s) => ({ messages: [...s.messages, m] })),

  updateLast: (patch) =>
    set((s) => {
      if (s.messages.length === 0) return s;
      const messages = s.messages.slice();
      messages[messages.length - 1] = {
        ...messages[messages.length - 1],
        ...patch,
      };
      return { messages };
    }),

  setStatus: (status) => set({ status }),
  setModel: (model) => set({ model }),
  setAnchors: ({ conversationId, lastResponseId }) =>
    set((s) => ({
      conversationId: conversationId ?? s.conversationId,
      lastResponseId: lastResponseId ?? s.lastResponseId,
    })),

  loadHistory: (detail) =>
    set({
      messages: normalizeHistoryMessages(detail),
      conversationId: detail.id,
      lastResponseId: detail.latest_response_id ?? undefined,
      status: "idle",
    }),

  reset: () =>
    set({
      messages: [],
      status: "idle",
      conversationId: undefined,
      lastResponseId: undefined,
    }),
}));
```

- [ ] **Step 4: Run to verify chat store passes**

Run: `cd newfrontend && npm test -- store/__tests__/chat`
Expected: PASS.

- [ ] **Step 5: Write the failing test for the conversations store** — `newfrontend/src/store/__tests__/conversations.test.ts`

```ts
import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../../api/conversations", () => ({
  listConversations: vi.fn(),
  deleteConversation: vi.fn(),
}));
vi.mock("../../lib/user", () => ({ getUserId: () => "u1" }));

import { useConversationsStore } from "../conversations";
import * as api from "../../api/conversations";

beforeEach(() => {
  vi.clearAllMocks();
  useConversationsStore.setState({ items: [], selectedId: undefined });
});

describe("conversations store", () => {
  it("refresh loads the list for the current user", async () => {
    (api.listConversations as any).mockResolvedValue([
      { id: "c1", title: "a", created_at: null, updated_at: null, last_response_id: "r1" },
    ]);
    await useConversationsStore.getState().refresh();
    expect(api.listConversations).toHaveBeenCalledWith("u1");
    expect(useConversationsStore.getState().items).toHaveLength(1);
  });

  it("select / clearSelection update selectedId", () => {
    useConversationsStore.getState().select("c1");
    expect(useConversationsStore.getState().selectedId).toBe("c1");
    useConversationsStore.getState().clearSelection();
    expect(useConversationsStore.getState().selectedId).toBeUndefined();
  });

  it("remove deletes via api and drops from the list", async () => {
    useConversationsStore.setState({
      items: [
        { id: "c1", title: "a", created_at: null, updated_at: null, last_response_id: "r1" },
      ],
      selectedId: "c1",
    });
    (api.deleteConversation as any).mockResolvedValue(undefined);
    await useConversationsStore.getState().remove("c1");
    expect(api.deleteConversation).toHaveBeenCalledWith("c1");
    expect(useConversationsStore.getState().items).toHaveLength(0);
    expect(useConversationsStore.getState().selectedId).toBeUndefined();
  });
});
```

- [ ] **Step 6: Run to verify failure**

Run: `cd newfrontend && npm test -- store/__tests__/conversations`
Expected: FAIL — cannot resolve `../conversations`.

- [ ] **Step 7: Implement `newfrontend/src/store/conversations.ts`**

```ts
import { create } from "zustand";
import type { ConversationSummary } from "../types";
import { listConversations, deleteConversation } from "../api/conversations";
import { getUserId } from "../lib/user";

interface ConversationsState {
  items: ConversationSummary[];
  selectedId?: string;
  refresh: () => Promise<void>;
  select: (id: string) => void;
  clearSelection: () => void;
  remove: (id: string) => Promise<void>;
}

export const useConversationsStore = create<ConversationsState>((set, get) => ({
  items: [],
  selectedId: undefined,

  refresh: async () => {
    const items = await listConversations(getUserId());
    set({ items });
  },

  select: (id) => set({ selectedId: id }),
  clearSelection: () => set({ selectedId: undefined }),

  remove: async (id) => {
    await deleteConversation(id);
    set((s) => ({
      items: s.items.filter((c) => c.id !== id),
      selectedId: s.selectedId === id ? undefined : s.selectedId,
    }));
  },
}));
```

- [ ] **Step 8: Run to verify pass + build**

Run: `cd newfrontend && npm test -- store && npm run build`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add newfrontend/src/store
git commit -m "feat(newfrontend): zustand chat + conversations stores with history normalization"
```

---

## Task 5: `useResponsesChat` hook (send / stop / regenerate)

The orchestration: owns the openai stream, drives the reducer into the chat store, and exposes send/stop/regenerate. The stream source is injected so it tests against a mocked async iterable.

**Files:**
- Create: `newfrontend/src/hooks/useResponsesChat.ts`
- Test: `newfrontend/src/hooks/__tests__/useResponsesChat.test.tsx`

**Interfaces:**
- Consumes: `useChatStore` (`../store/chat`); `streamResponse` (`../api/client`); `getUserId` (`../lib/user`); `initialStreamState`/`reduceStreamEvent` (`../stream/reducer`); `useConversationsStore.refresh` (`../store/conversations`).
- Produces: `useResponsesChat()` returning `{ send(text: string): Promise<void>; stop(): void; regenerate(): Promise<void>; isStreaming: boolean }`. The hook reads `streamResponse` through the module import (mockable via `vi.mock`). On `send`: append the user message + a streaming assistant placeholder; stream; on each event patch the last message via the reducer; on completion advance `conversationId`/`lastResponseId` and refresh the sidebar. On `stop`: abort; mark the last message `status:"stopped"`; do NOT advance anchors. On `regenerate`: re-send the last user input using the anchors that produced the turn being regenerated.

- [ ] **Step 1: Write the failing test** — `newfrontend/src/hooks/__tests__/useResponsesChat.test.tsx`

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";

vi.mock("../../api/client", () => ({ streamResponse: vi.fn() }));
vi.mock("../../lib/user", () => ({ getUserId: () => "u1" }));
vi.mock("../../store/conversations", () => ({
  useConversationsStore: { getState: () => ({ refresh: vi.fn() }) },
}));

import { useResponsesChat } from "../useResponsesChat";
import { useChatStore } from "../../store/chat";
import * as client from "../../api/client";

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

describe("useResponsesChat", () => {
  it("send streams a turn, captures text and advances anchors", async () => {
    (client.streamResponse as any).mockReturnValue(
      streamOf([
        { type: "response.created", response: { id: "resp_1", conversation: { id: "conv_1" } } },
        { type: "response.output_text.delta", delta: "hi" },
        {
          type: "response.completed",
          response: {
            id: "resp_1",
            conversation: { id: "conv_1" },
            status: "completed",
            usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
          },
        },
      ])
    );
    const { result } = renderHook(() => useResponsesChat());
    await act(async () => {
      await result.current.send("hello");
    });
    const st = useChatStore.getState();
    expect(st.messages.map((m) => m.role)).toEqual(["user", "assistant"]);
    expect(st.messages[1].text).toBe("hi");
    expect(st.messages[1].status).toBe("completed");
    expect(st.conversationId).toBe("conv_1");
    expect(st.lastResponseId).toBe("resp_1");
    // continuation: a second send carries the conversation anchor
    (client.streamResponse as any).mockReturnValue(
      streamOf([
        { type: "response.created", response: { id: "resp_2", conversation: { id: "conv_1" } } },
        { type: "response.completed", response: { id: "resp_2", conversation: { id: "conv_1" }, status: "completed" } },
      ])
    );
    await act(async () => {
      await result.current.send("again");
    });
    const params = (client.streamResponse as any).mock.calls[1][0];
    expect(params.conversation).toBe("conv_1");
    expect(params.previous_response_id).toBe("resp_1");
    expect(params.user_id).toBe("u1");
  });

  it("stop aborts and marks the bubble stopped without advancing anchors", async () => {
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => (release = r));
    (client.streamResponse as any).mockReturnValue({
      async *[Symbol.asyncIterator]() {
        yield { type: "response.created", response: { id: "resp_1", conversation: { id: "conv_1" } } };
        yield { type: "response.output_text.delta", delta: "partial" };
        await gate; // never resolves before stop()
      },
    });
    const { result } = renderHook(() => useResponsesChat());
    let sendPromise: Promise<void>;
    await act(async () => {
      sendPromise = result.current.send("hello");
      await Promise.resolve();
    });
    act(() => result.current.stop());
    release();
    await act(async () => {
      await sendPromise;
    });
    const st = useChatStore.getState();
    expect(st.messages[1].status).toBe("stopped");
    expect(st.messages[1].text).toBe("partial");
    // Stop does NOT advance anchors (first turn -> next send starts fresh)
    expect(st.conversationId).toBeUndefined();
    expect(st.lastResponseId).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd newfrontend && npm test -- useResponsesChat`
Expected: FAIL — cannot resolve `../useResponsesChat`.

- [ ] **Step 3: Implement `newfrontend/src/hooks/useResponsesChat.ts`**

```ts
import { useCallback, useRef, useState } from "react";
import { streamResponse } from "../api/client";
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

export function useResponsesChat() {
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const lastInputRef = useRef<string>("");

  const runTurn = useCallback(
    async (
      text: string,
      anchors: { conversation?: string; previousResponseId?: string }
    ) => {
      const chat = useChatStore.getState();
      lastInputRef.current = text;

      const userMsg: ChatMessage = {
        id: tempId("user"),
        role: "user",
        text,
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
      abortRef.current = controller;
      let state: StreamState = initialStreamState(assistantMsg.id);
      let aborted = false;

      try {
        const stream = streamResponse(
          {
            model: chat.model,
            input: text,
            user_id: getUserId(),
            conversation: anchors.conversation,
            previous_response_id: anchors.previousResponseId,
          },
          controller.signal
        );
        for await (const event of stream) {
          state = reduceStreamEvent(state, event);
          useChatStore.getState().updateLast({
            text: state.message.text,
            reasoning: state.message.reasoning,
            reasoningStatus: state.message.reasoningStatus,
            status: state.message.status,
            responseId: state.message.responseId,
            usage: state.message.usage,
            error: state.message.error,
          });
        }
      } catch (err) {
        if (controller.signal.aborted) {
          aborted = true;
        } else {
          useChatStore.getState().updateLast({
            status: "failed",
            error: err instanceof Error ? err.message : "stream error",
          });
        }
      } finally {
        abortRef.current = null;
        setIsStreaming(false);
        useChatStore.getState().setStatus("idle");
      }

      if (aborted) {
        // Stop is local-only: keep the partial bubble, do NOT advance anchors.
        useChatStore.getState().updateLast({ status: "stopped" });
        return;
      }

      if (state.message.status === "completed") {
        useChatStore.getState().setAnchors({
          conversationId: state.conversationId,
          lastResponseId: state.responseId,
        });
        void useConversationsStore.getState().refresh();
      }
    },
    []
  );

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      const chat = useChatStore.getState();
      await runTurn(trimmed, {
        conversation: chat.conversationId,
        previousResponseId: chat.lastResponseId,
      });
    },
    [runTurn]
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const regenerate = useCallback(async () => {
    const text = lastInputRef.current;
    if (!text) return;
    const chat = useChatStore.getState();
    // Re-send the last input with the anchors that produced the prior turn.
    await runTurn(text, {
      conversation: chat.conversationId,
      previousResponseId: chat.lastResponseId,
    });
  }, [runTurn]);

  return { send, stop, regenerate, isStreaming };
}
```

NOTE on the abort path: aborting the openai stream rejects the `for await` with an abort error; we detect it via `controller.signal.aborted` and convert the in-flight bubble to `status:"stopped"` WITHOUT calling `setAnchors`, so the next `send` continues from the last *persisted* response (or starts fresh if this was the first turn) — exactly the spec's Stop semantics.

- [ ] **Step 4: Run to verify pass**

Run: `cd newfrontend && npm test -- useResponsesChat`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the whole suite + build**

Run: `cd newfrontend && npm test && npm run build`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add newfrontend/src/hooks
git commit -m "feat(newfrontend): useResponsesChat hook (send/stop/regenerate over the stream reducer)"
```

---

## Task 6: Presentational components

The leaf components: Markdown rendering with code copy, the collapsible reasoning panel, message controls, the composer, the model selector, and the user/assistant bubbles. Component-tested with `@testing-library/react`.

**Files:**
- Create: `newfrontend/src/components/Markdown.tsx`, `CollapsibleReasoning.tsx`, `MessageControls.tsx`, `Composer.tsx`, `ModelSelector.tsx`, `UserMessage.tsx`, `AssistantMessage.tsx`
- Test: `newfrontend/src/components/__tests__/CollapsibleReasoning.test.tsx`, `Markdown.test.tsx`, `MessageControls.test.tsx`, `Composer.test.tsx`

**Interfaces:**
- Consumes: `ChatMessage` (`../types`); `cn` (`../lib/cn`); `useResponsesChat` is NOT used here (containers wire it in Task 7).
- Produces:
  - `Markdown({ content }: { content: string })`.
  - `CollapsibleReasoning({ reasoning, status }: { reasoning: string; status: ReasoningStatus })` — auto-open while `streaming`, auto-collapse on `done`; nothing rendered when reasoning is empty.
  - `MessageControls({ text, onRegenerate }: { text: string; onRegenerate?: () => void })` — copy + (optional) regenerate buttons.
  - `Composer({ onSend, onStop, isStreaming }: { onSend: (t: string) => void; onStop: () => void; isStreaming: boolean })`.
  - `ModelSelector({ model, onChange }: { model: string; onChange: (m: string) => void })`.
  - `UserMessage({ message })`, `AssistantMessage({ message, onRegenerate })`.

- [ ] **Step 1: Write the failing tests**

`newfrontend/src/components/__tests__/CollapsibleReasoning.test.tsx`:

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { CollapsibleReasoning } from "../CollapsibleReasoning";

describe("CollapsibleReasoning", () => {
  it("renders nothing when reasoning is empty", () => {
    const { container } = render(
      <CollapsibleReasoning reasoning="" status="idle" />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("shows reasoning text while streaming (expanded)", () => {
    render(<CollapsibleReasoning reasoning="thinking..." status="streaming" />);
    expect(screen.getByText("thinking...")).toBeVisible();
  });

  it("renders a toggle once done", () => {
    render(<CollapsibleReasoning reasoning="done thinking" status="done" />);
    expect(screen.getByRole("button")).toBeInTheDocument();
  });
});
```

`newfrontend/src/components/__tests__/Markdown.test.tsx`:

```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Markdown } from "../Markdown";

describe("Markdown", () => {
  it("renders headings and bold", () => {
    render(<Markdown content={"# Title\n\nsome **bold** text"} />);
    expect(screen.getByRole("heading", { name: "Title" })).toBeInTheDocument();
    expect(screen.getByText("bold")).toBeInTheDocument();
  });
});
```

`newfrontend/src/components/__tests__/MessageControls.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MessageControls } from "../MessageControls";

describe("MessageControls", () => {
  beforeEach(() => {
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
  });

  it("copies text to the clipboard", async () => {
    render(<MessageControls text="hello world" />);
    await userEvent.click(screen.getByRole("button", { name: /copy/i }));
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith("hello world");
  });

  it("calls onRegenerate when provided", async () => {
    const onRegenerate = vi.fn();
    render(<MessageControls text="x" onRegenerate={onRegenerate} />);
    await userEvent.click(screen.getByRole("button", { name: /regenerate/i }));
    expect(onRegenerate).toHaveBeenCalled();
  });
});
```

`newfrontend/src/components/__tests__/Composer.test.tsx`:

```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Composer } from "../Composer";

describe("Composer", () => {
  it("sends on click and clears the input", async () => {
    const onSend = vi.fn();
    render(<Composer onSend={onSend} onStop={() => {}} isStreaming={false} />);
    const box = screen.getByRole("textbox");
    await userEvent.type(box, "hi there");
    await userEvent.click(screen.getByRole("button", { name: /send/i }));
    expect(onSend).toHaveBeenCalledWith("hi there");
    expect((box as HTMLTextAreaElement).value).toBe("");
  });

  it("shows a stop button while streaming", async () => {
    const onStop = vi.fn();
    render(<Composer onSend={() => {}} onStop={onStop} isStreaming={true} />);
    await userEvent.click(screen.getByRole("button", { name: /stop/i }));
    expect(onStop).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd newfrontend && npm test -- components`
Expected: FAIL — cannot resolve the component modules.

- [ ] **Step 3: Implement `newfrontend/src/components/Markdown.tsx`**

```tsx
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

export function Markdown({ content }: { content: string }) {
  return (
    <div className="prose prose-sm max-w-none break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            const isBlock = Boolean(match);
            if (!isBlock) {
              return (
                <code className="rounded bg-gray-100 px-1 py-0.5" {...props}>
                  {children}
                </code>
              );
            }
            return (
              <SyntaxHighlighter
                language={match![1]}
                style={oneDark}
                PreTag="div"
              >
                {String(children).replace(/\n$/, "")}
              </SyntaxHighlighter>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
```

- [ ] **Step 4: Implement `newfrontend/src/components/CollapsibleReasoning.tsx`**

```tsx
import { useEffect, useState } from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { ChevronRight, Brain } from "lucide-react";
import type { ReasoningStatus } from "../types";
import { cn } from "../lib/cn";

export function CollapsibleReasoning({
  reasoning,
  status,
}: {
  reasoning: string;
  status: ReasoningStatus;
}) {
  const [open, setOpen] = useState(status === "streaming");

  // Auto-open while streaming, auto-collapse when reasoning finishes.
  useEffect(() => {
    if (status === "streaming") setOpen(true);
    else if (status === "done") setOpen(false);
  }, [status]);

  if (!reasoning) return null;

  return (
    <Collapsible.Root
      open={open}
      onOpenChange={setOpen}
      className="mb-2 rounded-md border border-gray-200 bg-gray-50 text-sm"
    >
      <Collapsible.Trigger className="flex w-full items-center gap-1 px-2 py-1 text-gray-500">
        <ChevronRight
          className={cn("h-4 w-4 transition-transform", open && "rotate-90")}
        />
        <Brain className="h-4 w-4" />
        <span>{status === "streaming" ? "Reasoning…" : "Reasoning"}</span>
      </Collapsible.Trigger>
      <Collapsible.Content className="whitespace-pre-wrap px-3 py-2 text-gray-600">
        {reasoning}
      </Collapsible.Content>
    </Collapsible.Root>
  );
}
```

- [ ] **Step 5: Implement `newfrontend/src/components/MessageControls.tsx`**

```tsx
import { Copy, RefreshCw, Check } from "lucide-react";
import { useState } from "react";

export function MessageControls({
  text,
  onRegenerate,
}: {
  text: string;
  onRegenerate?: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <div className="mt-1 flex gap-2 text-gray-400">
      <button
        type="button"
        aria-label="Copy"
        onClick={copy}
        className="hover:text-gray-700"
      >
        {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
      </button>
      {onRegenerate && (
        <button
          type="button"
          aria-label="Regenerate"
          onClick={onRegenerate}
          className="hover:text-gray-700"
        >
          <RefreshCw className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}
```

- [ ] **Step 6: Implement `newfrontend/src/components/Composer.tsx`**

```tsx
import { useState, type KeyboardEvent } from "react";
import { SendHorizontal, Square } from "lucide-react";

export function Composer({
  onSend,
  onStop,
  isStreaming,
}: {
  onSend: (text: string) => void;
  onStop: () => void;
  isStreaming: boolean;
}) {
  const [value, setValue] = useState("");

  const submit = () => {
    const text = value.trim();
    if (!text) return;
    onSend(text);
    setValue("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isStreaming) submit();
    }
  };

  return (
    <div className="flex items-end gap-2 border-t border-gray-200 p-3">
      <textarea
        className="min-h-[44px] flex-1 resize-none rounded-md border border-gray-300 p-2 outline-none focus:border-gray-500"
        placeholder="Send a message…"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKeyDown}
        rows={1}
      />
      {isStreaming ? (
        <button
          type="button"
          aria-label="Stop"
          onClick={onStop}
          className="rounded-md bg-gray-800 p-2 text-white"
        >
          <Square className="h-5 w-5" />
        </button>
      ) : (
        <button
          type="button"
          aria-label="Send"
          onClick={submit}
          className="rounded-md bg-blue-600 p-2 text-white disabled:opacity-50"
          disabled={!value.trim()}
        >
          <SendHorizontal className="h-5 w-5" />
        </button>
      )}
    </div>
  );
}
```

- [ ] **Step 7: Implement `newfrontend/src/components/ModelSelector.tsx`**

```tsx
const MODELS = ["gpt-4o-mini", "gpt-4o"];

export function ModelSelector({
  model,
  onChange,
}: {
  model: string;
  onChange: (m: string) => void;
}) {
  return (
    <select
      aria-label="Model"
      value={model}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-md border border-gray-300 px-2 py-1 text-sm"
    >
      {MODELS.map((m) => (
        <option key={m} value={m}>
          {m}
        </option>
      ))}
    </select>
  );
}
```

- [ ] **Step 8: Implement `newfrontend/src/components/UserMessage.tsx`**

```tsx
import type { ChatMessage } from "../types";

export function UserMessage({ message }: { message: ChatMessage }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl bg-blue-600 px-4 py-2 text-white">
        {message.text}
      </div>
    </div>
  );
}
```

- [ ] **Step 9: Implement `newfrontend/src/components/AssistantMessage.tsx`**

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
        {message.status === "completed" && (
          <MessageControls text={message.text} onRegenerate={onRegenerate} />
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 10: Run to verify pass**

Run: `cd newfrontend && npm test -- components`
Expected: PASS (all component tests).

- [ ] **Step 11: Build**

Run: `cd newfrontend && npm run build`
Expected: PASS.

- [ ] **Step 12: Commit**

```bash
git add newfrontend/src/components/Markdown.tsx newfrontend/src/components/CollapsibleReasoning.tsx newfrontend/src/components/MessageControls.tsx newfrontend/src/components/Composer.tsx newfrontend/src/components/ModelSelector.tsx newfrontend/src/components/UserMessage.tsx newfrontend/src/components/AssistantMessage.tsx newfrontend/src/components/__tests__
git commit -m "feat(newfrontend): presentational components (markdown, reasoning, controls, composer)"
```

---

## Task 7: Container components + app wiring

Assemble the shell: the sidebar (list/new/delete), the chat view (message list + composer + model selector), and the App that wires the stores + hook together. Replace the placeholder root.

**Files:**
- Create: `newfrontend/src/components/MessageList.tsx`, `Sidebar.tsx`, `ChatView.tsx`, `App.tsx`
- Modify: `newfrontend/src/main.tsx` (mount `<App/>` + `<Toaster/>`)
- Test: `newfrontend/src/components/__tests__/App.test.tsx`

**Interfaces:**
- Consumes: `useChatStore` (`../store/chat`), `useConversationsStore` (`../store/conversations`), `useResponsesChat` (`../hooks/useResponsesChat`), `getConversation` (`../api/conversations`), the presentational components, `sonner`'s `toast`/`Toaster`.
- Produces: `App()` (default-exported region rendered by `main.tsx`), `Sidebar()`, `ChatView()`, `MessageList()`.

- [ ] **Step 1: Write the failing test** — `newfrontend/src/components/__tests__/App.test.tsx`

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("../../api/conversations", () => ({
  listConversations: vi.fn().mockResolvedValue([]),
  getConversation: vi.fn(),
  deleteConversation: vi.fn(),
}));
vi.mock("../../api/client", () => ({ streamResponse: vi.fn() }));
vi.mock("../../lib/user", () => ({ getUserId: () => "u1" }));

import { App } from "../App";
import { useChatStore } from "../../store/chat";

beforeEach(() => useChatStore.getState().reset());

describe("App", () => {
  it("renders the composer and a New chat control", async () => {
    render(<App />);
    expect(screen.getByRole("textbox")).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: /new chat/i })
    ).toBeInTheDocument();
  });

  it("renders existing messages from the chat store", () => {
    useChatStore.setState({
      messages: [
        {
          id: "u",
          role: "user",
          text: "hello there",
          reasoning: "",
          reasoningStatus: "idle",
          status: "completed",
        },
      ],
    });
    render(<App />);
    expect(screen.getByText("hello there")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd newfrontend && npm test -- App`
Expected: FAIL — cannot resolve `../App`.

- [ ] **Step 3: Implement `newfrontend/src/components/MessageList.tsx`**

```tsx
import { useEffect, useRef } from "react";
import { useChatStore } from "../store/chat";
import { UserMessage } from "./UserMessage";
import { AssistantMessage } from "./AssistantMessage";

export function MessageList({ onRegenerate }: { onRegenerate: () => void }) {
  const messages = useChatStore((s) => s.messages);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const lastAssistantIndex = messages
    .map((m) => m.role)
    .lastIndexOf("assistant");

  return (
    <div className="flex-1 space-y-4 overflow-y-auto p-4">
      {messages.map((m, i) =>
        m.role === "user" ? (
          <UserMessage key={m.id} message={m} />
        ) : (
          <AssistantMessage
            key={m.id}
            message={m}
            onRegenerate={i === lastAssistantIndex ? onRegenerate : undefined}
          />
        )
      )}
      <div ref={bottomRef} />
    </div>
  );
}
```

- [ ] **Step 4: Implement `newfrontend/src/components/ChatView.tsx`**

```tsx
import { useChatStore } from "../store/chat";
import { useResponsesChat } from "../hooks/useResponsesChat";
import { MessageList } from "./MessageList";
import { Composer } from "./Composer";
import { ModelSelector } from "./ModelSelector";

export function ChatView() {
  const model = useChatStore((s) => s.model);
  const setModel = useChatStore((s) => s.setModel);
  const { send, stop, regenerate, isStreaming } = useResponsesChat();

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

- [ ] **Step 5: Implement `newfrontend/src/components/Sidebar.tsx`**

```tsx
import { useEffect } from "react";
import { Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { useConversationsStore } from "../store/conversations";
import { useChatStore } from "../store/chat";
import { getConversation } from "../api/conversations";
import { cn } from "../lib/cn";

export function Sidebar() {
  const items = useConversationsStore((s) => s.items);
  const selectedId = useConversationsStore((s) => s.selectedId);
  const refresh = useConversationsStore((s) => s.refresh);
  const select = useConversationsStore((s) => s.select);
  const clearSelection = useConversationsStore((s) => s.clearSelection);
  const remove = useConversationsStore((s) => s.remove);
  const loadHistory = useChatStore((s) => s.loadHistory);
  const reset = useChatStore((s) => s.reset);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const openConversation = async (id: string) => {
    try {
      const detail = await getConversation(id);
      loadHistory(detail);
      select(id);
    } catch {
      toast.error("Could not load conversation");
      clearSelection();
    }
  };

  const newChat = () => {
    reset();
    clearSelection();
  };

  const onDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await remove(id);
      if (selectedId === id) newChat();
    } catch {
      toast.error("Could not delete conversation");
    }
  };

  return (
    <aside className="flex w-64 flex-col border-r border-gray-200 bg-gray-50">
      <button
        type="button"
        onClick={newChat}
        className="m-2 flex items-center gap-2 rounded-md bg-blue-600 px-3 py-2 text-white"
      >
        <Plus className="h-4 w-4" /> New chat
      </button>
      <div className="flex-1 overflow-y-auto">
        {items.map((c) => (
          <div
            key={c.id}
            onClick={() => openConversation(c.id)}
            className={cn(
              "group flex cursor-pointer items-center justify-between px-3 py-2 text-sm hover:bg-gray-100",
              selectedId === c.id && "bg-gray-200"
            )}
          >
            <span className="truncate">{c.title || "Untitled"}</span>
            <button
              type="button"
              aria-label="Delete conversation"
              onClick={(e) => onDelete(e, c.id)}
              className="invisible text-gray-400 group-hover:visible hover:text-red-600"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
```

- [ ] **Step 6: Implement `newfrontend/src/components/App.tsx`**

```tsx
import { Sidebar } from "./Sidebar";
import { ChatView } from "./ChatView";

export function App() {
  return (
    <div className="flex h-full">
      <Sidebar />
      <ChatView />
    </div>
  );
}
```

- [ ] **Step 7: Update `newfrontend/src/main.tsx`**

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Toaster } from "sonner";
import { App } from "./components/App";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
    <Toaster position="top-center" />
  </StrictMode>
);
```

- [ ] **Step 8: Run to verify pass**

Run: `cd newfrontend && npm test -- App`
Expected: PASS (2 tests).

- [ ] **Step 9: Run the whole suite + build**

Run: `cd newfrontend && npm test && npm run build`
Expected: PASS (all tests; production build succeeds).

- [ ] **Step 10: Manual smoke (optional, requires the backend)**

Start the lean service (`cd backend && uvicorn app.lean_main:app --port 8000`), then `cd newfrontend && npm run dev`. Open the dev URL, send a message, confirm streaming text + collapsible reasoning, a new sidebar entry, stop/regenerate/copy, and reloading a conversation from the sidebar.

- [ ] **Step 11: Commit**

```bash
git add newfrontend/src/components/MessageList.tsx newfrontend/src/components/Sidebar.tsx newfrontend/src/components/ChatView.tsx newfrontend/src/components/App.tsx newfrontend/src/main.tsx newfrontend/src/components/__tests__/App.test.tsx
git commit -m "feat(newfrontend): app shell (sidebar + chat view) wired to stores + stream hook"
```

---

## Self-Review (completed against the spec)

- **Greenfield `newfrontend/` Vite + React + TS SPA, no Next.js/assistant-ui** → Task 1.
- **Stream `/v1/responses` via the openai JS SDK** (`responses.create({stream:true}, {signal})`) → `api/client.ts` (Task 2); consumed in the hook (Task 5).
- **Streaming text + Markdown** → `Markdown` + `AssistantMessage` (Task 6), driven by the reducer's `output_text.delta` handling (Task 3).
- **Collapsible streaming reasoning from the `reasoning.summary` events** → reducer handles `reasoning_summary_text.delta` / `..._part.done` (Task 3); `CollapsibleReasoning` auto-open/collapse (Task 6).
- **Multi-turn via `conversation` anchor + `previous_response_id`** → hook sends both, advances on completion (Task 5); continuation asserted in the hook test.
- **Conversation-history sidebar** → conversations store + `Sidebar` + `GET /v1/conversations[/{id}]` client (Tasks 2, 4, 7); history load via `loadHistory`/`normalizeHistoryMessages` (Task 4).
- **Stop / regenerate / copy** → Composer stop toggle, `MessageControls` (Task 6); hook `stop`/`regenerate` with local-only Stop semantics (no anchor advance, "stopped" bubble) (Task 5), asserted in tests.
- **Model selector** → `ModelSelector` + chat-store `model`/`setModel` (Tasks 4, 6, 7).
- **Per-browser `user_id` (localStorage UUID)** sent on responses + as `?user_id=` on list → `lib/user.ts` (Task 2), used in the hook + conversations store.
- **`dangerouslyAllowBrowser` + placeholder apiKey via Vite proxy** → `api/client.ts` (Task 2); proxy in `vite.config.ts` (Task 1).
- **Testing:** pure reducer against recorded sequences (plain text / reasoning+text / failed) + id capture (Task 3); continuation + Stop tests in the hook (Task 5); component tests for `CollapsibleReasoning`, `Markdown`, `MessageControls`, `Composer` (Task 6); conversations API client tests (Task 2). 
- **Out of scope honored:** no attachments, no tool/RAG/KB UI, no tenant auth/i18n/editing — `function_call*` events explicitly ignored in the reducer (Task 3) and tool UI omitted.

No placeholders; types are consistent across tasks (`ChatMessage`, `StreamState`, store action names, hook return shape, component props all match between definition and use).
