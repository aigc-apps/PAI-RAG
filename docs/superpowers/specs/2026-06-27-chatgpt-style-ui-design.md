# ChatGPT-style Agent UI + Tool/Reasoning Rendering — Design

**Date:** 2026-06-27
**Status:** Approved by delegation (user directive: modern minimal ChatGPT-style UI showing tools, thinking, answers, conversation list)
**Branch:** `personal/yfei/agent-core`
**Builds on:** `newfrontend/` (the SPA: reducer, stores, `useResponsesChat`, components) and `newbackend/` (the lean service: `responses_serializer`, `conversations_view`).

## Problem

The current `newfrontend/` works but is visually crude (raw Tailwind) and **does not show tool use at all**: the stream reducer drops every `function_call*` event, and the backend streams tool *calls* but never the tool *results* (only persists them; the conversation-history grouping also skips tool items). We want a **modern, minimal, ChatGPT-style** agent UI that correctly renders **reasoning (thinking), tool calls + results, the answer, and the conversation list** — live during streaming and on history reload.

## Goals

- A polished ChatGPT-like layout: collapsible sidebar (conversation list + new chat), centered conversation column, sticky composer, empty-state greeting.
- An assistant turn rendered as an **ordered timeline**: Thinking disclosure → tool cards → Markdown answer → hover controls.
- **Tools shown correctly** end-to-end: backend streams tool results and includes tool items in history; the reducer captures calls + results; the UI renders tool cards with status, args, and result.
- Keep all existing behavior (background runs, resume, cancel, user memory, model selector) working.

## Scope

**In scope:**
- **Backend (`newbackend`):** serializer emits a `response.tool_result` stream event on `ToolResult`; `conversations_view` attaches tool calls/results to the assistant history message; the detail route shape carries them.
- **Frontend (`newfrontend`):** `ChatMessage.toolCalls` + reducer capture of `function_call*` + `response.tool_result` (and history-loaded tool data); a **unified SSE stream client** (our `parseSSE` for the POST stream, replacing the OpenAI SDK, so custom events flow and one reader serves POST + resume); a ChatGPT-style design system + restyled/new components (Sidebar, ChatView/top bar, MessageList, UserMessage, AssistantMessage timeline, CollapsibleReasoning "Thinking", **ToolCall** card, Composer pill, Markdown, MessageControls, empty state).

**Out of scope (later):** dark theme (ship light first; structure with CSS vars so dark is additive); interleaved multi-step timeline ordering beyond reasoning→tools→answer; attachments/multimodal UI; auth; streaming-token virtualization.

## Decisions

1. **Unify on our own SSE reader for the POST stream.** Replace `client.responses.create({stream:true})` (OpenAI SDK) with `fetch('/v1/responses', {stream:true})` + `parseSSE` (already used by resume). The reducer already consumes plain event objects; this lets the backend emit a **custom `response.tool_result`** event without the SDK rejecting it, and removes the `openai` dep + `dangerouslyAllowBrowser` from the browser. Cancel still uses `POST .../cancel`; background unchanged.
2. **Tool result as a custom stream event.** `ToolResult` → `{type:"response.tool_result", call_id, output, ok, sequence_number}`. (OpenAI has no server-side tool-result *output* stream event; `function_call_output` is an input item.) History parity comes from the grouping change, not the SDK envelope.
3. **Assistant message carries an ordered `toolCalls` list**, rendered between the Thinking disclosure and the answer. v1 ordering is reasoning → tools → answer (good enough; true interleaving deferred).
4. **Light theme via CSS variables** in `index.css`; components use the tokens. ChatGPT-like neutrals.

## Architecture

### Backend

- **`responses_serializer.serialize_response_stream`** — in the `ToolResult` branch, after `asm.on_tool_result(...)`, also `yield _sse(ToolResultEvent(...))`. Since this is a custom (non-OpenAI) event, emit it as a hand-built SSE dict (not a pydantic SDK type): a small helper `_sse_obj({"type":"response.tool_result","call_id":...,"output":...,"ok":...,"sequence_number":nxt()})`. Keep `sequence_number` monotonic (so resume cursors stay correct). The sync serializer is unchanged.
- **`conversations_view.group_conversation_messages`** — for each turn, collect `function_call` items (`{call_id,name,arguments}`) and match `function_call_output` items (`{call_id,output}`); attach `tool_calls: [{call_id,name,arguments,output}]` to the assistant message dict. (Currently these items are skipped.) The detail route already serializes the assistant message dict, so it carries `tool_calls` automatically; document the new field.

### Frontend

- **Types (`types.ts`):** `ToolUse = { id: string; name: string; arguments: string; status: "running"|"done"|"error"; output?: string; error?: string }`; `ChatMessage.toolCalls: ToolUse[]` (default `[]`).
- **Reducer (`stream/reducer.ts`):** new cases —
  - `response.output_item.added` with `item.type==="function_call"` → push `{id:item.id||call_id, name:item.name, arguments:"", status:"running"}`.
  - `response.function_call_arguments.delta` → append `delta` to the matching call's `arguments`.
  - `response.function_call_arguments.done` → set `arguments`, `name`.
  - `response.output_item.done` with `item.type==="function_call"` → keep (status stays running until the result).
  - `response.tool_result` → set `output`/`error`, `status = ok ? "done" : "error"` on the call with `call_id`.
  - Capture `sequence_number` as today. Match calls by `call_id` (the function_call `item.id` is `fc_<call_id>`; the events carry `item_id`/`call_id` — match on the `call_id` suffix consistently).
- **History (`store/chat.ts normalizeHistoryMessages`):** map the detail message's `tool_calls` → `ToolUse[]` (status `"done"`, output present).
- **Stream client (`api/client.ts`):** `streamResponse(params, signal)` → `fetch("/v1/responses", {method:"POST", body: JSON.stringify({...params, store:true, stream:true}), signal, headers})` → `parseSSE(res.body)`. Drop the `openai` import. `useResponsesChat` unchanged (it already iterates the async iterable).

### Design system (`index.css` + components)

CSS variables (light): `--bg:#fff`, `--bg-sidebar:#f9f9f9`, `--text:#0d0d0d`, `--text-muted:#676767`, `--text-faint:#9b9b9b`, `--border:#e5e5e5`, `--user-bubble:#f4f4f4`, `--tool-bg:#f7f7f8`, `--accent:#0d0d0d`, `--accent-fg:#fff`, `--danger:#e02e2e`, `--radius:12px`. System font stack; 16px base; comfortable line-height.

Components (Tailwind v4, tokens via arbitrary values / a small set of `@theme` or utility classes):
- **App shell:** `flex h-full`; Sidebar + main column.
- **Sidebar:** ~260px, `--bg-sidebar`; top "New chat" (ghost + icon); conversation list (truncated title, hover/active bg, trash on hover); collapsible (a toggle that hides it on narrow screens / a hamburger). 
- **Top bar:** minimal — a `ModelSelector` (clean dropdown) and (optional) conversation title; subtle bottom border.
- **MessageList:** centered `max-w-3xl mx-auto px-4`, generous vertical spacing, autoscroll.
- **UserMessage:** right-aligned, `bg-[--user-bubble] rounded-3xl px-4 py-2.5 max-w-[80%]`.
- **AssistantMessage timeline:** small assistant glyph; then `CollapsibleReasoning` → each `ToolCall` → `Markdown` answer (or error box / cancelled note) → `MessageControls` on hover.
- **CollapsibleReasoning ("Thinking"):** a muted disclosure row (chevron + "Thinking…" while streaming / "Thought" when done) that auto-expands while streaming and auto-collapses on done; body in muted text.
- **ToolCall (new):** a bordered card (`bg-[--tool-bg] rounded-xl border`): header = tool icon (lucide `Wrench`) + name + a status indicator (spinner running / check done / x error); expandable details = arguments (mono, small) and the result/error (mono, scrollable, truncated with "show more").
- **Composer:** centered, `max-w-3xl`, a rounded-3xl bordered container with subtle shadow; auto-grow textarea; circular send button (lucide `ArrowUp`) dark, disabled when empty; square stop while streaming; Enter sends, Shift+Enter newline.
- **Empty state:** when no messages, center a greeting ("What can I help with?") with the composer mid-screen.
- **Markdown:** typographic styles (headings/lists/tables/links/blockquote) + fenced code with a copy button and syntax highlighting (keep `react-syntax-highlighter`).

## Data flow (assistant turn)

Stream events → reducer builds `{reasoning, reasoningStatus, toolCalls[], text, status}` on the in-flight message → `AssistantMessage` renders Thinking (from reasoning) → ToolCall cards (from toolCalls, in arrival order) → Markdown (from text). On reload, `normalizeHistoryMessages` reconstructs the same shape (reasoning + toolCalls with results + text) so history looks identical to live.

## Testing

- **Backend:** stream serializer emits a `response.tool_result` event (with `output`, `ok`, `sequence_number`) after a tool call; `group_conversation_messages` attaches `tool_calls` (call+result) to the assistant message; a reloaded conversation detail carries them.
- **Frontend reducer:** a tool turn (`output_item.added` fc → `function_call_arguments.delta/done` → `response.tool_result`) yields a `toolCalls` entry with name, accumulated arguments, output, and status `done`; an error result → status `error`; `sequence_number` still tracked; `normalizeHistoryMessages` maps history `tool_calls`.
- **Stream client:** `streamResponse` POSTs to `/v1/responses` with `stream:true` and yields parsed events from `parseSSE` (mock fetch with a ReadableStream); cancel/background unchanged.
- **Components:** `ToolCall` renders name + status (running spinner / done check / error) + expandable args/result; `CollapsibleReasoning` streaming vs done; `AssistantMessage` renders the timeline (reasoning + tools + answer) and the failed/cancelled states; `Composer` send/stop + Enter; `Sidebar` list/new/delete/active; `App` smoke (empty state + a seeded message). `Markdown` headings/code/copy.
- Build (`npm run build`) + full suite green.

## Risks
- **Custom `response.tool_result`** is non-OpenAI; mitigated by owning the SSE reader (decision 1). The reducer keys final tool status off the event, history off the grouping.
- **Tool-call ↔ result matching** by `call_id` must be consistent between the `function_call` item id (`fc_<call_id>`) and the `tool_result.call_id`; the reducer normalizes on `call_id`.
- **Visual quality is subjective** — the design targets ChatGPT-light; CSS variables make later theme/spacing tweaks cheap. Iterate after first render.
- **Dropping the OpenAI SDK** from the browser removes `dangerouslyAllowBrowser`; the cancel/resume/list clients already use `fetch`, so the SDK becomes unused and is removed from deps.

## Sequencing
One spec, one plan, subagent-driven: backend (tool-result stream + history grouping) → frontend data (reducer/types/stream-client) → frontend UI (design system + shell + message/tool/reasoning rendering).
