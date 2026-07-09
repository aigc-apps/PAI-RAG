from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from fastapi import Request
from agent.agent import Agent
from agent.budgeting import AgentMessageManager
from agent.soul import Soul, DEFAULT_SOUL
from agent.tools.registry import ToolRegistry
from app.runs import RunManager
from app.providers import ProviderRouter


@dataclass
class AppState:
    store: object  # ResponseStore
    llm: object  # has .astream(messages, tools)
    default_model: str
    context_window: int = 110000
    max_output_tokens: int = 8000
    soul: Soul = field(default_factory=lambda: DEFAULT_SOUL)
    registry: ToolRegistry = field(default_factory=ToolRegistry)
    runs: RunManager = field(default_factory=RunManager)
    router: Optional[ProviderRouter] = None
    knowledge: object = None
    jobs: object = None  # JobQueue — the durable background worker pool
    agent_config: object = None
    memory_enabled: bool = False
    memory_model: str = ""
    summary_enabled: bool = False
    summary_keep_recent: int = 20
    summary_batch: int = 20
    project_context: str = ""

    def make_agent(self, llm=None, context_window: Optional[int] = None,
                   max_output_tokens: Optional[int] = None) -> Agent:
        # Verified signature: Agent(llm, max_steps=..., budget: Optional[AgentMessageManager]=None).
        # Pass an explicit budget from AppState's window so fake/echo test LLMs (which
        # lack a `context_window` attr) don't get a zero-width budget that truncates input.
        return Agent(
            llm=llm if llm is not None else self.llm,
            budget=AgentMessageManager(
                context_window=context_window if context_window is not None else self.context_window,
                max_output_tokens=max_output_tokens if max_output_tokens is not None else self.max_output_tokens,
            ),
        )


def get_state(request: Request) -> AppState:
    return request.app.state.app_state


def reload_app_state(state: AppState, settings) -> None:
    """Re-read the agent config from disk and rebuild the runtime registry +
    agent_config on ``state`` so control-plane changes (skill install/enable) take
    effect for subsequent requests without a process restart.

    The rebuilt registry is wired with this same reloader as ``on_config_change``
    so control-plane tools (e.g. ``enable_skill_for_agent``) keep refreshing the
    live state after each mutation. Imported lazily to avoid an import cycle with
    ``app.agent_config`` / ``agent.tools.defaults``."""
    from app.agent_config import apply_runtime_status, load_agent_config
    from agent.tools.defaults import build_default_registry

    doc = load_agent_config(settings.config_path)
    state.registry = build_default_registry(
        settings,
        agent_config=doc,
        on_config_change=lambda: reload_app_state(state, settings),
        # Preserve knowledge_search across control-plane reloads by rebinding the
        # already-live KnowledgeService (created at boot in lean_main).
        knowledge_service=state.knowledge,
    )
    # Vector-store selection is global (knowledgebase.vectordb). The registry
    # rebuild above doesn't touch the live search engine, so rebuild it here from
    # the fresh doc — this is what makes a Settings-UI vectordb change apply
    # without a restart (PUT /v1/config funnels through here via _save_and_reload).
    if state.knowledge is not None and hasattr(state.knowledge, "rebuild_search_engine"):
        state.knowledge.rebuild_search_engine(doc.knowledgebase.vectordb)
    state.agent_config = apply_runtime_status(doc, settings, state.router)
