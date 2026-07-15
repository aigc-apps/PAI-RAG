# Assistant Reasoning and Tool Timeline Design

## Goal

Render and persist an assistant turn in the order the model and agent produced
it. `reasoning_content`, ordinary `content`, and tool calls are peer timeline
entries rather than placing all reasoning in a single block above every tool.

A turn may therefore appear as:

```text
reasoning → content → tool → reasoning → content → tool → final content
```

The same order must survive conversation reloads.

## Timeline Model

The assistant timeline uses the existing `steps` concept and adds reasoning:

```typescript
type AssistantStep =
  | { kind: "reasoning"; text: string }
  | { kind: "text"; text: string }
  | { kind: "tool"; id: string };
```

Steps are append-only in event order. Consecutive deltas of the same textual
kind merge into the open step. A different kind closes the current step:

- reasoning delta after reasoning appends to that reasoning step;
- content delta after content appends to that text step;
- a tool start appends a tool step;
- reasoning or content after a tool starts a new textual step;
- content after reasoning starts a new text step, even without a tool between
  them.

`message.reasoning` remains an aggregate compatibility field. `message.text`
continues to point to the trailing text run used as the final answer.

## Streaming Path

The backend already serializes reasoning, content, and function-call events in
their runtime order. The frontend stream reducer will record reasoning deltas in
`message.steps` as well as the aggregate `message.reasoning`. Tool and content
handling continue to append to the same steps array.

`deriveAssistantView` treats the final trailing text step as the answer body.
All preceding reasoning, content, and tool steps become the ordered activity
timeline. If a turn currently ends on reasoning or a tool, the answer body stays
empty until final content arrives.

`AgentActivity` renders the activity steps in array order. It does not render the
aggregate reasoning above steps when reasoning steps are present, preventing
duplication and top-of-panel updates.

## Persistence Path

No database schema migration is required. The existing assistant `message`
item's JSON content gains a `timeline` array containing the same three step
shapes.

The response assembler maintains this timeline while consuming agent events:

- `ReasoningDelta` appends/merges a reasoning entry;
- `TextDelta` appends/merges a text entry;
- `ToolStarted` appends a tool reference at the earliest visible tool event;
- `ToolCompleted` ensures the reference exists for event streams that omitted
  `ToolStarted`, without duplicating it.

At finalization, the assistant message item stores aggregate `text` plus
`timeline`. The existing aggregate reasoning item remains stored for model
history and old-client compatibility. Function-call and function-call-output
items remain unchanged.

The conversation view reads `timeline` from the assistant item and returns it as
`steps`. The frontend history normalizer restores those steps and resolves tool
ids against the returned tool-call records.

## Compatibility

Pre-change conversations do not contain `timeline`. They retain the current
legacy layout: aggregate reasoning first, then tools, with the full assistant
text as the body. No existing rows are rewritten.

Clients that ignore the new field continue to receive `text`, `reasoning`, and
`tool_calls`. Model-context reconstruction continues to use the existing stored
items, so timeline display metadata does not alter prompts.

## Content and Final Answer Rules

Ordinary model `content` before a tool is activity narration and stays adjacent
to that tool. The last text run after the final reasoning/tool step is the final
answer and renders outside the collapsible activity panel.

If a model emits only reasoning followed by content, reasoning appears in the
activity panel and content is the answer. If it emits content without tools or
reasoning, that content remains the ordinary answer and no activity panel is
shown.

## Error and Resume Behavior

Partial steps remain visible for stopped, cancelled, failed, or interrupted
streams. SSE replay uses the reducer's existing duplicate-tool protection and
sequence cursor. Replayed deltas continue the last matching step rather than
creating a duplicate run.

If persisted timeline data is malformed, the conversation view drops invalid
entries and falls back to aggregate fields rather than failing the conversation
request.

## Tests

Backend tests cover:

- assembler ordering for repeated reasoning/content/tool cycles;
- merging consecutive same-kind deltas;
- `ToolStarted` plus `ToolCompleted` producing one tool step;
- streams without `ToolStarted` still recording the tool;
- timeline persistence on the assistant message item;
- conversation history returning steps in stored order;
- malformed/missing timeline compatibility.

Frontend tests cover:

- reducer ordering and same-kind merging;
- reasoning after a tool creating a new reasoning step;
- `deriveAssistantView` preserving mixed activity order and extracting only the
  trailing text as the answer;
- `AgentActivity` rendering reasoning/content/tool steps in DOM order without a
  duplicate top reasoning block;
- history normalization restoring persisted steps;
- old history without steps retaining the legacy layout.
