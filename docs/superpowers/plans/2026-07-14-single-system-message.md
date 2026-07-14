# Single System Message Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit exactly one system message containing the base prompt, environment, and volatile context, with no synthetic runtime-context message.

**Architecture:** Preserve `AgentContext.system_prompt` and `AgentContext.context_block` as separate construction-time fields. Merge their rendered text only in `Agent.build_messages()`, where the provider-facing message list is assembled.

**Tech Stack:** Python 3, pytest, existing `AgentContext` and `Message` models.

## Global Constraints

- Preserve history order and current-turn rendering.
- Omit a blank or whitespace-only context block.
- Do not change memory, summary, or additional-instruction rendering.
- Do not introduce another system or developer message.

---

## File Structure

- `backend/tests/test_agent_messages.py`: Specifies provider-facing message roles, content, and ordering.
- `backend/agent/agent.py`: Assembles the provider-facing message list from `AgentContext`.

### Task 1: Merge volatile context into the single system message

**Files:**
- Modify: `backend/tests/test_agent_messages.py`
- Modify: `backend/agent/agent.py`

**Interfaces:**
- Consumes: `Agent.build_messages(ctx: AgentContext) -> List[Message]`, `AgentContext.system_prompt`, and `AgentContext.context_block`.
- Produces: A message list whose first and only system message contains base prompt, environment, and optional context; history and current turn follow without a synthetic runtime-context message.

- [x] **Step 1: Replace the old runtime-context test with the required single-message behavior**

Update the first test in `backend/tests/test_agent_messages.py` to:

```python
def test_build_messages_merges_runtime_context_into_single_system_message():
    messages = Agent.build_messages(
        _context(context_block="# Memory\nThe user prefers concise answers.")
    )

    system_messages = [
        message for message in messages if message.role == "system"
    ]
    assert len(system_messages) == 1
    system = system_messages[0].content
    assert system.startswith("# Persona\nBe helpful.")
    assert "# Environment" in system
    assert "Today's date: 2026-07-14" in system
    assert "Time zone: Asia/Shanghai" in system
    assert "12:34:56" not in system
    assert "# Memory\nThe user prefers concise answers." in system
    assert system.index("# Environment") < system.index("# Memory")

    assert [(message.role, message.content) for message in messages[1:]] == [
        ("user", "earlier question"),
        ("assistant", "earlier answer"),
        ("user", "current question"),
    ]
    assert all(
        "<system-reminder>" not in str(message.content) for message in messages
    )
```

Add this adjacent blank-context regression test:

```python
def test_build_messages_omits_blank_runtime_context():
    messages = Agent.build_messages(_context(context_block="  \n"))

    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0].content.endswith("Time zone: Asia/Shanghai")
```

- [x] **Step 2: Run the focused tests and verify the new behavior fails**

Run:

```bash
cd backend && uv run pytest tests/test_agent_messages.py -q
```

Expected: the first test fails because `# Memory` is absent from the system message and a synthetic user message is still present. The blank-context test and unrelated tests may pass.

- [x] **Step 3: Implement the minimal single-system-message assembly**

Delete `_render_runtime_context()` from `backend/agent/agent.py`. Replace the final assembly portion of `Agent.build_messages()` with:

```python
system_parts = [
    ctx.system_prompt.strip(),
    environment,
]
if ctx.context_block.strip():
    system_parts.append(ctx.context_block.strip())

msgs: List[Message] = [Message("system", "\n\n".join(system_parts))]
msgs += ctx.history
msgs.append(render_current_turn(ctx.current_turn, ctx.attachments, ctx.hints))
```

- [x] **Step 4: Run the focused tests and verify they pass**

Run:

```bash
cd backend && uv run pytest tests/test_agent_messages.py -q
```

Expected: all tests in `tests/test_agent_messages.py` pass with no warnings or errors.

- [x] **Step 5: Run message-construction regression coverage**

Run:

```bash
cd backend && uv run pytest tests/test_agent_messages.py tests/test_builder.py tests/test_routes_responses.py tests/test_routes_conversations.py tests/test_lean_main_boot.py -q
```

Expected: all selected tests pass with no failures.

- [x] **Step 6: Check formatting and the final diff**

Run:

```bash
git diff --check
git diff -- backend/agent/agent.py backend/tests/test_agent_messages.py
```

Expected: `git diff --check` exits successfully; the diff contains only the message-assembly change and its tests.

- [x] **Step 7: Commit the implementation**

```bash
git add backend/agent/agent.py backend/tests/test_agent_messages.py docs/superpowers/plans/2026-07-14-single-system-message.md
git commit -m "fix(agent): merge runtime context into system prompt"
```
