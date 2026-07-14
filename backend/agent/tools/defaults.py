from __future__ import annotations
from typing import Any, Callable, Optional
from loguru import logger
from agent.tools.registry import ToolRegistry
from agent.tools.builtin.datetime_tool import make_current_datetime_tool
from agent.tools.builtin.web_fetch import make_web_fetch_tool
from agent.tools.builtin.web_search import make_web_search_tool, SearchProvider
from agent.tools.builtin.code_interpreter import make_code_interpreter_tool
from agent.tools.builtin.shell import make_shell_tool
from agent.tools.builtin.publish_artifact import make_publish_artifact_tool
from agent.tools.builtin.install_skill import make_install_skill_tool
from agent.tools.builtin.enable_skill import make_enable_skill_for_agent_tool
from agent.tools.builtin.load_skill import make_load_skill_tool
from agent.tools.builtin.knowledge import make_knowledge_search_tool
from agent.tools.builtin.knowledge_read import make_knowledge_read_tool
from agent.tools.builtin.knowledge_find import make_knowledge_find_tool
from agent.tools.builtin.knowledge_list import make_knowledge_list_tool
from agent.tools.search_providers import make_search_provider
from agent.tools.sandbox_providers import make_sandbox_provider
from agent.custom_skills import discover_skill_packages, skill_sources


def build_default_registry(
    settings,
    *,
    search_provider: Optional[SearchProvider] = None,
    agent_config=None,
    on_config_change: Optional[Callable[[], Any]] = None,
    knowledge_service=None,
) -> ToolRegistry:
    """Assemble the default registry. current_datetime + web_fetch always; web_search
    only when a provider is injected or `settings.search_provider != "none"`;
    knowledge tools only when a live KnowledgeService is passed in and the
    knowledge capability is not explicitly disabled."""
    reg = ToolRegistry()
    reg.register(make_current_datetime_tool())
    reg.register(make_web_fetch_tool())

    # Register knowledge as one complete capability. A missing control-plane config
    # means "not explicitly disabled", preserving isolated-test and alternate-host
    # behavior; an explicit disabled capability removes the entire bundle.
    if knowledge_service is not None and _knowledge_capability_available(agent_config):
        reg.register(make_knowledge_search_tool(knowledge_service))
        reg.register(make_knowledge_read_tool(knowledge_service))
        reg.register(make_knowledge_find_tool(knowledge_service))
        reg.register(make_knowledge_list_tool(knowledge_service))

    provider = search_provider
    if provider is None:
        provider = make_search_provider(agent_config)
    if provider is None and getattr(settings, "search_provider", "none") != "none":
        provider = _provider_from_settings(settings)
    if provider is not None:
        reg.register(make_web_search_tool(provider))

    sandbox_provider = make_sandbox_provider(agent_config)
    # Stash the warm provider on the registry so the /v1/files serve endpoint can
    # read artifact bytes back from the live sandbox when no backend NAS mount is
    # configured (best-effort fallback path).
    reg.sandbox_provider = sandbox_provider
    if sandbox_provider is not None:
        # The sandbox is an MCP-like provider that exposes several tools over one
        # warm instance: code_interpreter (run_code) and shell (run_command) share
        # the same mounts/env contract. Both are excludable per-agent via
        # tools.exclude.
        reg.register(
            make_code_interpreter_tool(
                sandbox_provider,
                default_timeout=sandbox_provider.default_timeout_seconds,
            )
        )
        reg.register(
            make_shell_tool(
                sandbox_provider,
                default_timeout=sandbox_provider.default_timeout_seconds,
            )
        )
        # publish_artifact surfaces sandbox files to the frontend. Only useful
        # when the signing secret is configured (else it would refuse at call
        # time); gate registration on it to keep the tool list clean.
        if getattr(settings, "files_url_secret", ""):
            reg.register(make_publish_artifact_tool(sandbox_provider, settings))
    # Progressive-disclosure skill loading: the always-injected catalog shows only
    # summaries and load_skill pulls SKILL.md on demand. Bundled files are accessed
    # only through the skill's read-only sandbox mount. Gate on at least one package
    # actually being
    # discovered — a configured-but-empty skills dir must NOT expose load_skill,
    # or the model, seeing the tool with an empty catalog, will hallucinate a
    # skill id (e.g. "skill.frontend-design") and call it. No skills → no tool.
    if agent_config is not None and discover_skill_packages(
        skill_sources(getattr(agent_config, "skills", None))
    ):
        reg.register(make_load_skill_tool())
    if _capability_enabled(agent_config, "install_skill"):
        reg.register(make_install_skill_tool(settings, agent_config))
    if _capability_enabled(agent_config, "enable_skill_for_agent"):
        reg.register(
            make_enable_skill_for_agent_tool(settings, agent_config, on_config_change=on_config_change)
        )
    names = reg.names()
    logger.info(
        "tool registry built: {} tools [{}] | publish_artifact={} (files_url_secret={}, sandbox={})",
        len(names),
        ", ".join(sorted(names)),
        "publish_artifact" in names,
        bool(getattr(settings, "files_url_secret", "")),
        reg.sandbox_provider is not None,
    )
    return reg


def _provider_from_settings(settings) -> Optional[SearchProvider]:
    """Hook for a real search backend (Tavily/Serp/etc.). Not wired to a vendor
    this iteration — returns None so web_search stays off unless a provider is
    injected explicitly. See the design's out-of-scope."""
    logger.info(
        f"search_provider={getattr(settings, 'search_provider', 'none')} configured "
        "but no live provider is wired yet; web_search disabled."
    )
    return None


def _capability_enabled(agent_config, capability_id: str) -> bool:
    """A capability is available unless explicitly permission="disabled". The old
    global ``enabled`` boolean is a deprecated no-op — availability is driven by
    provider presence + permission, and per-agent use by tools.include/exclude."""
    if agent_config is None:
        return False
    for cap in getattr(agent_config, "capabilities", []) or []:
        if cap.id == capability_id:
            return getattr(cap, "permission", "") != "disabled"
    return False


def _knowledge_capability_available(agent_config) -> bool:
    """Return false only when the control plane explicitly disables knowledge
    (permission="disabled"). A missing capability means "not disabled"."""
    if agent_config is None:
        return True
    for cap in getattr(agent_config, "capabilities", []) or []:
        if cap.id == "knowledge":
            return getattr(cap, "permission", "") != "disabled"
    return True
