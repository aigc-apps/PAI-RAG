"""spawn_subagent — delegate a self-contained sub-task to a subagent that runs in
an ISOLATED context and returns only a compact summary (the "context firewall").
This tool lives in ``agent/`` and never imports ``app/``: the concrete ``runner``
(which knows how to look up agent profiles, pick a model, build a child context and
drive the loop) is injected by the host wiring (``app.subagent.wire_subagents``).
The handler pulls the ambient ``ToolScope`` so the child runs under the same user —
identical to how ``load_skill`` reads its mounts.

Parallel fan-out (several subagents at once) is NOT a second tool: the model just
emits multiple spawn_subagent calls in one turn, and the agent loop dispatches
independent calls concurrently (see agent.agent._dispatch_parallel). One tool, no
single-vs-batch ambiguity for the model to resolve."""
from __future__ import annotations

from loguru import logger

from agent.tools.base import Tool
from agent.tools.scope import get_current_tool_scope

# Name the parent-context builder strips from a child's toolbox so a subagent can
# never spawn another (depth cap = 1). Imported by app.builder.build_subagent_context.
SPAWN_TOOL_NAMES = ("spawn_subagent",)


def _current_depth() -> int:
    try:
        return int(get_current_tool_scope().metadata.get("subagent_depth", 0))
    except Exception:
        return 0


def make_spawn_subagent_tool(runner) -> Tool:
    """`runner.run(agent_id, task, scope, depth) -> SubagentResult` is injected by
    the host (app.subagent.SubagentRunner). Duck-typed to keep this module free of
    any app import."""

    async def fn(agent_id: str, task: str) -> str:
        scope = get_current_tool_scope()
        res = await runner.run(
            agent_id=str(agent_id), task=str(task), scope=scope, depth=_current_depth()
        )
        if not res.ok:
            # Return the error as a normal string — never raise into the loop.
            return f"[subagent '{agent_id}' failed] {res.error}"
        logger.info(
            "[subagent] {} finished usage_total={} finish={}",
            agent_id, res.usage.total, res.finish_reason,
        )
        return res.summary or "(subagent returned no content)"

    return Tool(
        name="spawn_subagent",
        description=(
            "Delegate a self-contained sub-task to a subagent that runs in an ISOLATED "
            "context and returns ONLY a compact summary. Use it for noisy, multi-step "
            "work that should not clutter the main conversation — deep knowledge-base "
            "search, reading many documents, exploring the code. The subagent cannot "
            "see this conversation, so make `task` fully self-contained. Pass "
            "agent_id='explore' for the built-in read-only research worker (knowledge "
            "base + code + web). To fan out — e.g. search the knowledge base and the "
            "code from several angles at once — emit MULTIPLE spawn_subagent calls in "
            "the same turn; independent calls run in parallel."
        ),
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "A configured agent id, or 'explore' for the built-in research worker.",
                },
                "task": {
                    "type": "string",
                    "description": "Complete, self-contained instructions; the subagent sees only this, not the chat history.",
                },
            },
            "required": ["agent_id", "task"],
        },
        fn=fn,
    )
