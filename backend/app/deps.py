from __future__ import annotations
from dataclasses import dataclass, field
from time import monotonic
from typing import Optional
from fastapi import Request
from agent.agent import Agent
from agent.budgeting import AgentMessageManager
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
    config_store: object = None
    config_revision: int = 0
    config_checked_at: float = 0.0

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


async def get_state(request: Request) -> AppState:
    # async so FastAPI resolves it on the event loop directly; a sync `def`
    # dependency is dispatched through anyio's bounded threadpool on every
    # request, needlessly consuming a token under load.
    state = request.app.state.app_state
    await refresh_config_if_changed(state)
    return state


async def refresh_config_if_changed(state: AppState) -> None:
    if getattr(state, "config_store", None) is None:
        return
    from app.config import get_settings

    settings = get_settings()
    interval = max(float(getattr(settings, "config_reload_interval_seconds", 2.0)), 0.0)
    now = monotonic()
    if interval and now - state.config_checked_at < interval:
        return
    state.config_checked_at = now
    revision = await state.config_store.current_revision()
    if revision <= 0 or revision == state.config_revision:
        return
    stored = await state.config_store.load()
    from app.providers import ModelCatalog

    if state.router is not None:
        state.router.reload(ModelCatalog(**stored.doc.models))
    state.config_revision = stored.revision
    rebuild_app_state_from_config(state, settings, stored.doc)


def rebuild_app_state_from_config(state: AppState, settings, doc) -> None:
    """Rebuild runtime registry + agent_config from an authored config document.

    The rebuilt registry is wired with this same reloader as ``on_config_change``
    so control-plane tools (e.g. ``enable_skill_for_agent``) keep refreshing the
    live state after each mutation. Imported lazily to avoid an import cycle with
    ``app.agent_config`` / ``agent.tools.defaults``."""
    from app.agent_config import apply_runtime_status
    from agent.tools.defaults import build_default_registry

    async def _reload_from_store():
        if getattr(state, "config_store", None) is None:
            reload_app_state(state, settings)
            return
        stored = await state.config_store.load()
        from app.providers import ModelCatalog

        if state.router is not None:
            state.router.reload(ModelCatalog(**stored.doc.models))
        state.config_revision = stored.revision
        rebuild_app_state_from_config(state, settings, stored.doc)

    state.registry = build_default_registry(
        settings,
        agent_config=doc,
        on_config_change=_reload_from_store,
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
    # Register the spawn_subagent tools into the freshly rebuilt registry, bound to
    # a runner over this AppState (needs registry + agent_config, both set above).
    from app.subagent import wire_subagents
    wire_subagents(state)
    # read_handle recovers tool results the budget offloads from the window; bound
    # to the store so earlier-run results stay recoverable across stateless workers.
    from app.context_tools import wire_context_tools
    wire_context_tools(state)


def reload_app_state(state: AppState, settings) -> None:
    """Compatibility file-backed reloader for tests and non-SQL call sites."""
    from app.agent_config import load_agent_config

    rebuild_app_state_from_config(state, settings, load_agent_config(settings.config_path))
