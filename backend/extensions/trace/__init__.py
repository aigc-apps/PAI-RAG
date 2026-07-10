"""Optional OpenTelemetry tracing extension for the agent runtime.

The public API is exposed lazily via module ``__getattr__`` so that importing the
package itself pulls in **no** opentelemetry dependency — only the specific
symbol you access does (and only the ones backed by the SDK need it). This keeps
``extensions.trace.config`` / ``extensions.trace.semconv`` (pure stdlib) usable
in a lean deployment, and lets the core reference the heavy hooks through a
try/except without the package import fanning out to the whole SDK.

    from extensions.trace import init_tracing, shutdown_tracing, instrument_fastapi
    from extensions.trace import get_tracer, is_enabled
    from extensions.trace import use_current_span, pai_agent_wrapper
"""
from __future__ import annotations

_LAZY = {
    "init_tracing": ("setup", "init_tracing"),
    "shutdown_tracing": ("setup", "shutdown_tracing"),
    "instrument_fastapi": ("setup", "instrument_fastapi"),
    "get_tracer": ("tracer", "get_tracer"),
    "is_enabled": ("tracer", "is_enabled"),
    "use_current_span": ("base", "use_current_span"),
    "pai_agent_wrapper": ("pai_agent_wrapper", "pai_agent_wrapper"),
    "TraceConfig": ("config", "TraceConfig"),
}

__all__ = list(_LAZY)


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(f"{__name__}.{target[0]}")
    return getattr(module, target[1])
