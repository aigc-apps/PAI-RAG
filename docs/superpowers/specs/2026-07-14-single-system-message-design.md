# Single System Message Design

**Date:** 2026-07-14

## Goal

Send exactly one system message to the model. Volatile runtime context must be
part of that system message instead of being represented as a separate user or
system message.

## Context

`AgentContext` intentionally separates the stable `system_prompt` from the
volatile `context_block`. Today, `Agent.build_messages()` wraps the context
block in `<system-reminder>` and appends it as a synthetic user message just
before the current user turn. That synthetic message is not user-authored and
misrepresents the instruction hierarchy.

Provider APIs also do not share a portable multiple-system-message contract:
OpenAI accepts system/developer message items, while Anthropic and Gemini expose
a single top-level system instruction. A single assembled system message is the
most interoperable representation.

## Design

Keep `AgentContext.system_prompt` and `AgentContext.context_block` as separate
internal fields. At the model boundary, `Agent.build_messages()` assembles the
single system message in this order:

1. The configured base system prompt.
2. The generated environment section.
3. The volatile context block, when non-blank.

The context block remains structurally rendered by its existing producer; it is
trimmed before concatenation and is omitted when blank. The message list then
contains conversation history followed directly by the rendered current user
turn. No `<system-reminder>` wrapper or synthetic user message remains.

This preserves the stable-first token prefix for provider prompt caching while
keeping the public wire representation portable. It also keeps context
construction concerns separate from final message serialization.

## Error Handling and Safety

No new failure mode is introduced. Empty or whitespace-only context remains
omitted. Because the context block is no longer embedded inside an XML-like
wrapper, the special closing-tag replacement used by the old renderer is
removed with that renderer.

## Testing

Update the focused message-construction test first so the current implementation
fails for the intended reason. The test will assert that:

- exactly one system message is emitted;
- memory/runtime context is included in that system message after the
  environment section;
- no synthetic `<system-reminder>` user message exists;
- history ordering and the current user turn remain unchanged;
- blank context still produces the normal message sequence.

After the focused red-green cycle, run the relevant backend tests to catch
assumptions elsewhere about message roles or ordering.

## Non-goals

- Changing how memories, summaries, or additional instructions are rendered.
- Collapsing the two internal `AgentContext` fields into one.
- Changing persisted conversation history.
- Introducing multiple system or developer messages.
