from __future__ import annotations
from typing import Optional
from loguru import logger
from agent.tools.registry import ToolRegistry
from agent.tools.builtin.datetime_tool import make_current_datetime_tool
from agent.tools.builtin.web_fetch import make_web_fetch_tool
from agent.tools.builtin.web_search import make_web_search_tool, SearchProvider


def build_default_registry(
    settings, *, search_provider: Optional[SearchProvider] = None
) -> ToolRegistry:
    """Assemble the default registry. current_datetime + web_fetch always; web_search
    only when a provider is injected or `settings.search_provider != "none"`."""
    reg = ToolRegistry()
    reg.register(make_current_datetime_tool())
    reg.register(make_web_fetch_tool())

    provider = search_provider
    if provider is None and getattr(settings, "search_provider", "none") != "none":
        provider = _provider_from_settings(settings)
    if provider is not None:
        reg.register(make_web_search_tool(provider))
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
