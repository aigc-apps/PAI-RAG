# Assistant Reasoning and Tool Timeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render and restore `reasoning_content`, ordinary assistant content, and tool calls in their original execution order, while preserving the existing aggregate fields for compatibility.

**Architecture:** The responses serializer records a compact ordered `timeline` inside the persisted assistant message content. The conversation view validates and exposes it as `steps`; the frontend stream reducer produces the same structure live, and the assistant view renders reasoning/text/tool steps in order while treating only the trailing text step as the final answer. Existing `reasoning`, `text`, and tool-call records remain unchanged so old conversations and model replay continue to work.

**Tech Stack:** Python/FastAPI/Pydantic/OpenAI Responses protocol, React 19, TypeScript, Zustand, Vitest, pytest.

---

### Task 1: Persist an ordered backend timeline

**Files:**
- Modify: `backend/tests/test_responses_serializer_sync.py`
- Modify: `backend/api/protocol/responses_serializer.py`

- [ ] **Step 1: Write failing serializer tests**

Add a test whose event sequence is reasoning → text → tool start/completion → reasoning → final text, and assert that the persisted assistant message contains:

```python
[
    {"kind": "reasoning", "text": "先分析"},
    {"kind": "text", "text": "准备调用"},
    {"kind": "tool", "id": "c1"},
    {"kind": "reasoning", "text": "检查结果"},
    {"kind": "text", "text": "最终答案"},
]
```

Also assert consecutive deltas of one textual kind merge, `ToolStarted` plus `ToolCompleted` creates one tool step, and a `ToolCompleted` without a preceding start still creates one tool step.

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `cd backend && uv run pytest tests/test_responses_serializer_sync.py -q`

Expected: assertions fail because assistant message content has only `text` and no `timeline`.

- [ ] **Step 3: Implement the minimal assembler timeline**

In `_Assembler`, add a timeline and seen-tool-id set. Route deltas through a helper that merges only adjacent steps of the same textual kind:

```python
def _append_text_step(self, kind: str, text: str) -> None:
    if not text:
        return
    if self.timeline and self.timeline[-1].get("kind") == kind:
        self.timeline[-1]["text"] += text
    else:
        self.timeline.append({"kind": kind, "text": text})
```

Record a tool step on `ToolStarted`; make `on_tool_completed` call the same idempotent helper as a fallback. Handle `ToolStarted` in both sync and streaming serializers, including the streaming fallback path where arguments or completion arrive first. Persist `timeline` alongside `text` in the assistant message content without changing OpenAI response output ordering or legacy aggregate store items.

- [ ] **Step 4: Run the serializer tests and confirm GREEN**

Run: `cd backend && uv run pytest tests/test_responses_serializer_sync.py tests/test_responses_serializer_stream.py -q`

Expected: all pass.

- [ ] **Step 5: Commit the backend serializer change**

```bash
git add backend/api/protocol/responses_serializer.py backend/tests/test_responses_serializer_sync.py
git commit -m "feat: persist assistant execution timeline"
```

### Task 2: Expose validated timeline steps in conversation history

**Files:**
- Modify: `backend/tests/test_conversations_view.py`
- Modify: `backend/app/conversations_view.py`

- [ ] **Step 1: Write failing conversation-view tests**

Test that a valid assistant-message `timeline` is returned as `steps` unchanged. Add malformed cases (unknown kind, missing/non-string `text` or `id`, non-list timeline) and assert the response omits/falls back from `steps` rather than exposing invalid data. Confirm a legacy message with no timeline keeps its current shape and behavior.

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `cd backend && uv run pytest tests/test_conversations_view.py -q`

Expected: valid timelines are not surfaced yet.

- [ ] **Step 3: Implement strict timeline validation and exposure**

Add a small validator accepting only:

```python
{"kind": "reasoning", "text": "..."}
{"kind": "text", "text": "..."}
{"kind": "tool", "id": "call-id"}
```

Return `None` for a malformed timeline so the frontend uses the legacy layout. Include `steps` only when a valid, non-empty timeline was persisted. Do not infer order from aggregate reasoning or function-call items.

- [ ] **Step 4: Run backend history and serializer tests**

Run: `cd backend && uv run pytest tests/test_conversations_view.py tests/test_responses_serializer_sync.py tests/test_responses_serializer_stream.py -q`

Expected: all pass.

- [ ] **Step 5: Commit the history API change**

```bash
git add backend/app/conversations_view.py backend/tests/test_conversations_view.py
git commit -m "feat: restore assistant execution timeline"
```

### Task 3: Build the same ordered timeline during live streaming

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/stream/__tests__/reducer.test.ts`
- Modify: `frontend/src/stream/reducer.ts`

- [ ] **Step 1: Write failing reducer tests**

Extend `AssistantStep` expectations to cover reasoning → tool → reasoning → final text. Assert adjacent reasoning deltas merge, reasoning following a tool creates a new reasoning step, replayed tool-added events do not duplicate either `toolCalls` or timeline steps, and aggregate `message.reasoning` remains available.

Add a resume/legacy-state test proving that when aggregate reasoning/text exists without steps, the reducer seeds compatible steps before appending a new event rather than discarding existing content.

- [ ] **Step 2: Run the focused reducer test and confirm RED**

Run: `cd frontend && npm test -- src/stream/__tests__/reducer.test.ts`

Expected: TypeScript or timeline assertions fail because reasoning is not an `AssistantStep`.

- [ ] **Step 3: Implement reasoning steps and safe seeding**

Extend the union:

```ts
export type AssistantStep =
  | { kind: "reasoning"; text: string }
  | { kind: "text"; text: string }
  | { kind: "tool"; id: string };
```

Use a shared helper to seed missing steps from legacy aggregate reasoning/text, append or merge adjacent reasoning/text deltas, and add a tool id only once. Keep `message.reasoning` and the existing trailing `message.text` semantics unchanged.

- [ ] **Step 4: Run the reducer test and confirm GREEN**

Run: `cd frontend && npm test -- src/stream/__tests__/reducer.test.ts`

Expected: all pass.

- [ ] **Step 5: Commit live timeline support**

```bash
git add frontend/src/types.ts frontend/src/stream/reducer.ts frontend/src/stream/__tests__/reducer.test.ts
git commit -m "feat: interleave reasoning in live timeline"
```

### Task 4: Hydrate and render history in execution order

**Files:**
- Modify: `frontend/src/store/__tests__/chat.test.ts`
- Modify: `frontend/src/store/chat.ts`
- Modify: `frontend/src/stream/__tests__/assistantView.test.ts`
- Modify: `frontend/src/stream/assistantView.ts`
- Modify: `frontend/src/components/AgentActivity.tsx`
- Test or create: `frontend/src/components/__tests__/AgentActivity.test.tsx`

- [ ] **Step 1: Write failing hydration and view tests**

Assert history normalization copies backend `steps` into `ChatMessage`. In `deriveAssistantView`, assert reasoning/tool/reasoning/text order is preserved and the trailing text alone becomes `bodyText`. Add a component test checking DOM text order and proving aggregate reasoning is not rendered a second time when timeline reasoning exists; keep a test for aggregate-only legacy reasoning.

- [ ] **Step 2: Run focused frontend tests and confirm RED**

Run: `cd frontend && npm test -- src/store/__tests__/chat.test.ts src/stream/__tests__/assistantView.test.ts src/components/__tests__/AgentActivity.test.tsx`

Expected: history drops steps and the view/component cannot resolve reasoning steps.

- [ ] **Step 3: Implement hydration and ordered rendering**

Add optional `steps` to the wire history type and copy validated arrays into the normalized message. Extend `ResolvedStep` with reasoning. Resolve reasoning and narration separately while continuing to drop whitespace-only activity blocks. In `AgentActivity`, derive `hasReasoning` from timeline or the legacy aggregate, render the aggregate reasoning only if there is no timeline reasoning, and render every reasoning/text/tool activity step in array order.

- [ ] **Step 4: Run focused frontend tests and confirm GREEN**

Run: `cd frontend && npm test -- src/store/__tests__/chat.test.ts src/stream/__tests__/assistantView.test.ts src/components/__tests__/AgentActivity.test.tsx`

Expected: all pass.

- [ ] **Step 5: Commit history rendering support**

```bash
git add frontend/src/store/chat.ts frontend/src/store/__tests__/chat.test.ts frontend/src/stream/assistantView.ts frontend/src/stream/__tests__/assistantView.test.ts frontend/src/components/AgentActivity.tsx frontend/src/components/__tests__/AgentActivity.test.tsx
git commit -m "feat: render assistant execution timeline in order"
```

### Task 5: Full verification and delivery

**Files:**
- Review: all modified files

- [ ] **Step 1: Run backend regression tests**

Run: `cd backend && uv run pytest tests/test_responses_serializer_sync.py tests/test_responses_serializer_stream.py tests/test_conversations_view.py -q`

Expected: all pass.

- [ ] **Step 2: Run the frontend suite and production build**

Run: `cd frontend && npm test`

Expected: all tests pass.

Run: `cd frontend && npm run build`

Expected: TypeScript and Vite production build succeed.

- [ ] **Step 3: Inspect the final diff and worktree**

Run: `git diff --check && git status --short --branch && git log --oneline -6`

Expected: no whitespace errors; only intended committed changes; branch ahead of the remote by the new commits.

- [ ] **Step 4: Push the current branch**

Run: `git push origin personal/yfei/agent-core`

Expected: remote branch advances successfully.
