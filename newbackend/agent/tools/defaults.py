from __future__ import annotations
from typing import Any, Callable, Optional
from loguru import logger
from agent.tools.registry import ToolRegistry
from agent.tools.builtin.datetime_tool import make_current_datetime_tool
from agent.tools.builtin.web_fetch import make_web_fetch_tool
from agent.tools.builtin.web_search import make_web_search_tool, SearchProvider
from agent.tools.builtin.code_interpreter import make_code_interpreter_tool
from agent.tools.builtin.shell import make_shell_tool
from agent.tools.builtin.install_skill import make_install_skill_tool
from agent.tools.builtin.enable_skill import make_enable_skill_for_agent_tool
from agent.tools.builtin.load_skill import make_load_skill_tool
from agent.tools.builtin.read_skill_resource import make_read_skill_resource_tool
from agent.tools.search_providers import make_search_provider
from agent.tools.sandbox_providers import make_sandbox_provider
from agent.custom_skills import skill_sources


def build_default_registry(
    settings,
    *,
    search_provider: Optional[SearchProvider] = None,
    agent_config=None,
    on_config_change: Optional[Callable[[], Any]] = None,
) -> ToolRegistry:
    """Assemble the default registry. current_datetime + web_fetch always; web_search
    only when a provider is injected or `settings.search_provider != "none"`."""
    reg = ToolRegistry()
    reg.register(make_current_datetime_tool())
    reg.register(make_web_fetch_tool())

    provider = search_provider
    if provider is None:
        provider = make_search_provider(agent_config)
    if provider is None and getattr(settings, "search_provider", "none") != "none":
        provider = _provider_from_settings(settings)
    if provider is not None:
        reg.register(make_web_search_tool(provider))

    sandbox_provider = make_sandbox_provider(agent_config)
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
    # Progressive-disclosure skill loading (read-only, safe): the always-injected
    # catalog shows only summaries; load_skill pulls a skill's full instructions on
    # demand and read_skill_resource reads its bundled files (host-side, path-jailed,
    # no sandbox needed). Registered whenever any skill source is configured.
    if agent_config is not None and skill_sources(getattr(agent_config, "skills", None)):
        reg.register(make_load_skill_tool())
        reg.register(make_read_skill_resource_tool())
    if _capability_enabled(agent_config, "install_skill"):
        reg.register(make_install_skill_tool(settings, agent_config))
    if _capability_enabled(agent_config, "enable_skill_for_agent"):
        reg.register(
            make_enable_skill_for_agent_tool(settings, agent_config, on_config_change=on_config_change)
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
    if agent_config is None:
        return False
    for cap in getattr(agent_config, "capabilities", []) or []:
        if cap.id == capability_id:
            return bool(cap.enabled and cap.permission != "disabled")
    return False
