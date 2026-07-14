"""Subagent orchestration — the host side of the context firewall.

``SubagentRunner`` turns "run agent X on task T in an isolated window" into a single
call: look up the agent profile (a configured ``AgentProfile`` or the built-in
``explore`` worker), pick its model, build a clean child ``AgentContext`` (empty
history, no memory/summary), drive ``Agent.run`` to completion, and collapse the
stream into a compact summary + usage.

Design notes (see docs/superpowers/specs/2026-07-12-spawn-subagent-design.md):
- The subagent's *capability envelope* is predefined and governable (a profile);
  only the *task* is generated at runtime by the parent. No dynamic system prompts.
- The runner holds ``AppState`` and reads registry/router/config lazily at call
  time, so config hot-reloads (which mutate the same AppState) are picked up.
- Every failure is returned as ``SubagentResult(ok=False, error=...)`` — never
  raised — so a subagent can't crash the parent loop.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from agent.core.events import RunCompleted, RunFailed, TextDelta, Usage
from agent.tools.scope import ToolScope
from app.agent_config import AgentProfile, AgentToolsConfig
from app.builder import build_subagent_context
from utils.constants import try_get_int_env

# Hard ceilings. A subagent is a full LLM run; keep it bounded so fan-out can't run
# away. max_steps is smaller than a top-level run (its job is focused).
SUBAGENT_TIMEOUT_SECONDS = try_get_int_env("SUBAGENT_TIMEOUT_SECONDS", 240)
SUBAGENT_MAX_STEPS = try_get_int_env("SUBAGENT_MAX_STEPS", 12)
MAX_SUBAGENT_DEPTH = try_get_int_env("MAX_SUBAGENT_DEPTH", 1)


@dataclass
class SubagentResult:
    ok: bool
    summary: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    error: Optional[str] = None


# ---- Built-in "explore" worker ------------------------------------------------
# A read-only research subagent tuned for the headline scenario: code exploration
# and knowledge-base search. Synthesised at call time (never stored in config) so
# operators get it for free. `include` is intersected with whatever tools are
# actually registered, so a missing sandbox/KB just drops the relevant tools.
_EXPLORE_INSTRUCTIONS = """\
You are an exploration and research subagent. Dig through the available sources and \
come back with a focused, well-evidenced answer.

- Before answering about product behavior, configuration, procedures, APIs, error \
messages or codes, troubleshooting, policies, or other potentially documented facts, \
call knowledge_search first. Do not search for greetings, identity questions, casual \
conversation, or pure writing or translation tasks unless documented facts are needed; \
when forming a query, retain product or service names and exact error identifiers from \
the request and conversation (for example, `TurboX license_check 失败`); \
use knowledge_list only when discovery or narrowing is useful. When a retrieved \
passage is incomplete, ambiguous, or lacks context, call knowledge_read with its \
internal document_id or chunk_id. Use knowledge_find for exact identifiers, error \
codes, API names, or literal phrases. For each knowledge-base citation, always show \
the document title; add an accessible HTTP or HTTPS URL when present, otherwise show \
the title only; never expose internal document_id or chunk_id to the user.
- For questions about how this system's own code behaves, if the knowledge base comes \
up short and a read-only code layer is available at /opt/code, explore it with the \
shell tool (ripgrep/grep to find symbols, cat/sed to read files) and cite file paths \
with line ranges.
- Use web_search / web_fetch only for current or external facts the knowledge base and \
code cannot answer.

Report the answer with its evidence, and say plainly when something could not be found."""

_EXPLORE_TOOLS = [
    "knowledge_search", "knowledge_read", "knowledge_find", "knowledge_list",
    "web_search", "web_fetch", "shell", "code_interpreter", "current_datetime",
]


def _explore_profile() -> AgentProfile:
    return AgentProfile(
        id="explore",
        name="Explore",
        description="Built-in read-only research worker: knowledge base + code + web.",
        instructions=_EXPLORE_INSTRUCTIONS,
        tools=AgentToolsConfig(include=list(_EXPLORE_TOOLS)),
        settings={"max_steps": SUBAGENT_MAX_STEPS},
    )


_BUILTIN_PROFILES = {"explore": _explore_profile}


class SubagentRunner:
    def __init__(self, state) -> None:
        self._state = state

    def _resolve_profile(self, agent_id: str) -> Optional[AgentProfile]:
        builtin = _BUILTIN_PROFILES.get(agent_id)
        if builtin is not None:
            return builtin()
        agents = getattr(self._state.agent_config, "agents", None) or []
        return next((a for a in agents if a.id == agent_id), None)

    def _resolve_llm(self, profile):
        """Return (llm, context_window, max_output_tokens) for the child's model.
        Prefer the router (per-agent pinned model or the deployment default);
        fall back to AppState's single client when no router is configured."""
        model_id = (getattr(profile, "model", "") or "") or self._state.default_model
        router = getattr(self._state, "router", None)
        if router is not None:
            try:
                cfg = router.get_config(model_id)
                return router.get_llm(model_id), cfg.context_window, cfg.max_output_tokens
            except KeyError:
                pass  # unknown model id — fall through to the default client
        if self._state.llm is None:
            raise RuntimeError(f"no LLM available for model '{model_id}'")
        return (
            self._state.llm,
            getattr(self._state, "context_window", 0),
            getattr(self._state, "max_output_tokens", 0),
        )

    async def run(self, *, agent_id: str, task: str, scope: ToolScope,
                  depth: int = 0) -> SubagentResult:
        if depth >= MAX_SUBAGENT_DEPTH:
            return SubagentResult(False, "", error="nested subagents are not allowed")
        if not (task or "").strip():
            return SubagentResult(False, "", error="task is empty")

        profile = self._resolve_profile(agent_id)
        if profile is None:
            known = ", ".join(a.id for a in (getattr(self._state.agent_config, "agents", None) or []))
            return SubagentResult(
                False, "",
                error=f"unknown agent_id '{agent_id}'; available: {known or '(none)'}, explore",
            )

        try:
            llm, context_window, max_output_tokens = self._resolve_llm(profile)
        except Exception as exc:  # noqa: BLE001 — surfaced as an error result, never raised
            return SubagentResult(False, "", error=f"model unavailable: {exc}")

        ctx = build_subagent_context(
            profile=profile,
            task=task,
            parent_scope=scope,
            registry=self._state.registry,
            agent_config=self._state.agent_config,
            project_context=getattr(self._state, "project_context", "") or "",
            depth=depth + 1,
        )
        agent = self._state.make_agent(
            llm=llm, context_window=context_window, max_output_tokens=max_output_tokens,
        )
        agent.max_steps = _child_max_steps(profile, agent.max_steps)
        logger.info(
            "[subagent] spawning id={} model={} tools={} depth={}",
            profile.id, getattr(profile, "model", "") or self._state.default_model,
            len(ctx.tools.tools), depth + 1,
        )
        return await _collect_run(agent, ctx, timeout=SUBAGENT_TIMEOUT_SECONDS)


def _child_max_steps(profile, parent_default: int) -> int:
    requested = int((getattr(profile, "settings", None) or {}).get("max_steps", SUBAGENT_MAX_STEPS))
    return max(1, min(requested, SUBAGENT_MAX_STEPS, parent_default))


async def _collect_run(agent, ctx, *, timeout: int) -> SubagentResult:
    """Drive a child Agent.run to completion, collapsing its event stream into a
    summary string + usage. Wrapped in a hard timeout."""

    async def _drive():
        parts: list[str] = []
        usage = Usage()
        finish = "stop"
        error: Optional[str] = None
        events = await agent.run(ctx)
        async for ev in events:
            if isinstance(ev, TextDelta):
                parts.append(ev.text)
            elif isinstance(ev, RunCompleted):
                usage = ev.usage
                finish = ev.finish_reason
            elif isinstance(ev, RunFailed):
                error = ev.message
                finish = "error"
        return "".join(parts).strip(), usage, finish, error

    try:
        summary, usage, finish, error = await asyncio.wait_for(_drive(), timeout=timeout)
    except asyncio.TimeoutError:
        return SubagentResult(False, "", finish_reason="timeout",
                              error=f"subagent exceeded {timeout}s")
    if error is not None:
        return SubagentResult(False, summary, usage=usage, finish_reason=finish, error=error)
    return SubagentResult(True, summary, usage=usage, finish_reason=finish)


# ---- Host wiring --------------------------------------------------------------
def subagent_enabled(agent_config) -> bool:
    """Gate registration on the ``subagent`` capability. Absent (older configs) =>
    on: the explore worker is a safe, read-only-by-default context tool and this is
    a headline capability. Individual agents still opt in via ``tools.include``."""
    for cap in (getattr(agent_config, "capabilities", None) or []):
        if getattr(cap, "id", "") == "subagent":
            return bool(getattr(cap, "enabled", False))
    return True


def wire_subagents(state) -> None:
    """Register the spawn tools into ``state.registry``, bound to a runner over
    ``state``. Called after the registry + agent_config are in place — at boot and
    on every config reload — because the runner needs the fully-assembled AppState."""
    if state.registry is None or not subagent_enabled(state.agent_config):
        return
    # Imported here (not at module top) to keep the import graph clean: this app-side
    # module reaches into the agent-side tool factory only when actually wiring.
    from agent.tools.builtin.spawn_subagent import make_spawn_subagent_tool

    runner = SubagentRunner(state)
    state.registry.register(make_spawn_subagent_tool(runner))
