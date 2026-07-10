"""Accessors for the process-wide tracer.

Importing this module requires ``opentelemetry`` to be installed; callers in the
lean core guard the import with try/except and fall back to a no-op. Once
``init_tracing`` has run, ``get_tracer()`` returns a tracer bound to the global
provider; before that (or when tracing is disabled) it still returns a valid
tracer whose spans go to the default no-op provider — cheap and side-effect-free.
"""
from __future__ import annotations

from opentelemetry import trace

_INSTRUMENTATION_NAME = "pai.agent"
_ENABLED = False


def get_tracer():
    return trace.get_tracer(_INSTRUMENTATION_NAME)


def is_enabled() -> bool:
    """True once ``init_tracing`` has successfully wired an exporter."""
    return _ENABLED


def _set_enabled(value: bool) -> None:
    global _ENABLED
    _ENABLED = value
