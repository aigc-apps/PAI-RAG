"""Tiered context management: offload-not-truncate + read_handle recovery.

The budget compressor must replace an over-budget tool result with a compact
placeholder that is *losslessly recoverable* — never the old lossy head+tail cut.
read_handle then recovers the full body from the in-run cache (same run, not yet
persisted) or the durable store (earlier runs), scoped to the caller's conversation.
"""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.budgeting import AgentMessageManager, MessageGroup
from agent.context_offload import is_placeholder
from agent.tools.builtin.read_handle import make_read_handle_tool
from agent.tools.scope import ToolScope, set_current_tool_scope, reset_current_tool_scope
from app.store.memory import InMemoryStore
from app.store.base import Item


def _mgr():
    return AgentMessageManager(context_window=110000, max_output_tokens=8000)


def _tool_round(call_id: str, output: str) -> MessageGroup:
    assistant = {"role": "assistant", "content": None,
                 "tool_calls": [{"id": call_id, "type": "function",
                                 "function": {"name": "grep_file", "arguments": "{}"}}]}
    tool = {"role": "tool", "tool_call_id": call_id, "content": output}
    return MessageGroup(messages=[assistant, tool], tokens=0, group_type="tool_round")


# --------------------------------------------------------------------------- #
# offload-not-truncate: the L1 pass replaces a big result with a recoverable
# placeholder instead of discarding bytes.
# --------------------------------------------------------------------------- #
def test_truncate_group_offloads_to_recoverable_placeholder():
    mgr = _mgr()
    mgr.run_bodies = {}
    full = "LINE-" + ("payload " * 1000)          # ~8KB, well over the 200-token gate
    group = _tool_round("call_abc", full)

    saved = mgr._truncate_tool_results_in_group(group)
    tool_msg = group.messages[1]
    # Window copy is now a compact placeholder, not the raw body...
    assert is_placeholder(tool_msg["content"])
    assert "read_handle" in tool_msg["content"]
    assert len(tool_msg["content"]) < len(full)
    assert saved > 0
    # ...but the full body is losslessly recoverable for this run.
    assert mgr.run_bodies["call_abc"] == full


def test_placeholder_is_not_reoffloaded():
    mgr = _mgr()
    mgr.run_bodies = {}
    full = "payload " * 1000
    group = _tool_round("call_abc", full)
    mgr._truncate_tool_results_in_group(group)
    placeholder = group.messages[1]["content"]
    # A second compression pass must be a no-op on an already-offloaded result.
    saved2 = mgr._truncate_tool_results_in_group(group)
    assert saved2 == 0
    assert group.messages[1]["content"] == placeholder


def test_small_tool_result_is_left_inline():
    mgr = _mgr()
    mgr.run_bodies = {}
    small = "ok"
    group = _tool_round("call_small", small)
    saved = mgr._truncate_tool_results_in_group(group)
    assert saved == 0
    assert group.messages[1]["content"] == small
    assert "call_small" not in mgr.run_bodies


def test_fit_to_budget_offloads_older_rounds_losslessly():
    mgr = _mgr()
    # Budget chosen so L1 offload of the older rounds alone gets under budget — the
    # realistic path. (min_protected=0 unprotects the earlier rounds; the newest
    # tool round + system stay protected.) A far tinier budget would push on to L2
    # summarization/L3 drop, which is a separate, lossier tier.
    mgr.min_protected_history_rounds = 0
    mgr.token_budget = 1500
    mgr.run_bodies = {}
    bodies = {f"call_{i}": ("row " * 800 + f"#{i}") for i in range(3)}
    messages = [{"role": "system", "content": "sys"}]
    for cid, out in bodies.items():
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": cid, "type": "function",
                                         "function": {"name": "grep_file", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": cid, "content": out})
    messages.append({"role": "user", "content": "what did we find?"})

    fitted = mgr.fit_to_budget(messages)
    placeholders = [m for m in fitted
                    if m.get("role") == "tool" and is_placeholder(m.get("content", ""))]
    assert placeholders, "expected at least one offloaded tool result"
    # Every offloaded round is recoverable verbatim from the in-run bodies map.
    for m in placeholders:
        cid = m["tool_call_id"]
        assert mgr.run_bodies[cid] == bodies[cid]


# --------------------------------------------------------------------------- #
# read_handle: recover from the in-run cache, then the store; conversation-scoped.
# --------------------------------------------------------------------------- #
def _call(tool, scope, **kwargs):
    async def go():
        t = set_current_tool_scope(scope)
        try:
            return await tool.fn(**kwargs)
        finally:
            reset_current_tool_scope(t)
    return asyncio.run(go())


def test_read_handle_recovers_from_in_run_bodies():
    tool = make_read_handle_tool(resolver=None)   # no store; the run map is the only source
    full = "the full body\nline two\nline three"
    scope = ToolScope(conversation_id="c1", run_bodies={"call_x": full})

    async def go():
        t = set_current_tool_scope(scope)
        try:
            got = await tool.fn(handle="store://tool/call_x")
            ranged = await tool.fn(handle="call_x", start=1, count=1)
        finally:
            reset_current_tool_scope(t)
        return got, ranged

    got, ranged = asyncio.run(go())
    assert got == full
    assert ranged == "line two"


async def _call_async(tool, scope, **kwargs):
    t = set_current_tool_scope(scope)
    try:
        return await tool.fn(**kwargs)
    finally:
        reset_current_tool_scope(t)


def test_read_handle_recovers_from_store_across_runs():
    async def go():
        store = InMemoryStore()
        conv = await store.create_conversation(user_id="u1")
        await store.append_items(conv.id, [
            Item(type="function_call_output",
                 content={"call_id": "call_hist", "output": "historical full output"}),
        ])
        tool = make_read_handle_tool(resolver=store)
        # No in-run cache entry → falls through to the store, scoped to the conv.
        return await _call_async(tool, ToolScope(conversation_id=conv.id),
                                 handle="store://tool/call_hist")
    assert asyncio.run(go()) == "historical full output"


def test_read_handle_is_conversation_scoped():
    async def go():
        store = InMemoryStore()
        a = await store.create_conversation(user_id="u1")
        await store.append_items(a.id, [
            Item(type="function_call_output",
                 content={"call_id": "call_secret", "output": "conv-A private data"}),
        ])
        tool = make_read_handle_tool(resolver=store)
        # A different conversation's scope must NOT resolve conv-A's result.
        return await _call_async(tool, ToolScope(conversation_id="other-conv"),
                                 handle="call_secret")
    out = asyncio.run(go())
    assert "no stored content" in out


def test_read_handle_rejects_empty_handle():
    tool = make_read_handle_tool(resolver=None)
    out = _call(tool, ToolScope(conversation_id="c1"), handle="")
    assert "valid handle is required" in out
