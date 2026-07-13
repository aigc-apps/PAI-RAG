"""Host wiring for tiered-context tools.

``read_handle`` needs the ResponseStore to recover offloaded tool results from
earlier runs, so — like ``spawn_subagent`` — it is registered here (not in the
store-agnostic ``build_default_registry``), after the registry + store are in place.
Called at boot and on every config reload, because a registry rebuild drops the tool.
"""
from __future__ import annotations


def wire_context_tools(state) -> None:
    """Register ``read_handle`` into ``state.registry``, bound to ``state.store`` as
    the durable resolver. No-op without a registry."""
    if getattr(state, "registry", None) is None:
        return
    from agent.tools.builtin.read_handle import make_read_handle_tool

    state.registry.register(make_read_handle_tool(getattr(state, "store", None)))
