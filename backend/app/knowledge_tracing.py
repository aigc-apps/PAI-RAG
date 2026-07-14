"""Optional, privacy-safe tracing helpers for online knowledge search."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any, Iterator, Optional


def _get_tracer():
    from extensions.trace.tracer import get_tracer

    return get_tracer()


@contextmanager
def knowledge_span(
    name: str, attributes: Optional[dict[str, Any]] = None
) -> Iterator[Any | None]:
    """Start a span when tracing is available, otherwise yield ``None``."""
    with ExitStack() as stack:
        try:
            span = stack.enter_context(_get_tracer().start_as_current_span(name))
        except Exception:
            yield None
            return
        set_span_attributes(span, attributes or {})
        yield span


def set_span_attributes(span, attributes: dict[str, Any]) -> None:
    if span is None:
        return
    for key, value in attributes.items():
        if value is None:
            continue
        try:
            span.set_attribute(key, value)
        except Exception:
            pass


def add_span_event(
    span, name: str, attributes: Optional[dict[str, Any]] = None
) -> None:
    if span is None:
        return
    try:
        span.add_event(name, attributes=attributes or {})
    except Exception:
        pass


def mark_span_error(span, error: Exception) -> None:
    """Mark failure without recording the exception message or payload."""
    set_span_attributes(
        span,
        {"error.type": type(error).__name__, "knowledge.status": "error"},
    )
    if span is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR))
    except Exception:
        pass
