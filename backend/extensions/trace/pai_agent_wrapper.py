"""Root span around the whole agent run.

Decorates ``Agent.run`` (an ``async def`` that returns an async generator). We
open one ``AGENT``-kind span that stays open for the entire lifetime of the
returned generator, so every LLM generation and tool call the loop produces
nests underneath it — yielding the Langfuse trace tree

    agent.run (AGENT)
      ├─ ChatCompletion (LLM, auto-instrumented)
      └─ tool <name>   (TOOL, manual)

Attributes are filled from the ``AgentContext`` up front (input, user, session,
model) and back-filled from the streamed events (visible output text, token
usage, terminal status). Falls back to the untouched generator when tracing is
absent or disabled, so the lean core is unaffected.
"""
from __future__ import annotations

from functools import wraps
from typing import Any

from . import semconv as sc


def _turn_text(turn: Any) -> str:
    """Best-effort plain-text of the current user turn (str or multimodal list)."""
    content = getattr(turn, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content
                 if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(t for t in parts if t)
    return "" if content is None else str(content)


def _set_input(span, self, ctx) -> None:
    try:
        span.set_attribute(sc.OPENINFERENCE_SPAN_KIND, sc.SpanKind.AGENT)
        text = _turn_text(getattr(ctx, "current_turn", None))
        if text:
            span.set_attribute(sc.INPUT_VALUE, text)
            span.set_attribute(sc.INPUT_MIME_TYPE, sc.MIME_TEXT)
        if getattr(ctx, "user_id", None):
            span.set_attribute(sc.USER_ID, str(ctx.user_id))
        if getattr(ctx, "conversation_id", None):
            span.set_attribute(sc.SESSION_ID, str(ctx.conversation_id))
        if getattr(ctx, "agent_id", None):
            span.set_attribute("agent.id", str(ctx.agent_id))
        model = getattr(getattr(self, "llm", None), "model", None)
        if model:
            span.set_attribute(sc.LLM_MODEL_NAME, str(model))
    except Exception:
        pass


def _capture_event(span, ev, buf: list) -> None:
    """Accumulate output text and back-fill terminal attributes from events."""
    etype = getattr(ev, "type", "")
    try:
        if etype == "text.delta":
            buf.append(getattr(ev, "text", "") or "")
        elif etype == "run.completed":
            usage = getattr(ev, "usage", None)
            if usage is not None:
                span.set_attribute(sc.LLM_TOKEN_COUNT_PROMPT, int(getattr(usage, "input", 0) or 0))
                span.set_attribute(sc.LLM_TOKEN_COUNT_COMPLETION, int(getattr(usage, "output", 0) or 0))
                span.set_attribute(sc.LLM_TOKEN_COUNT_TOTAL, int(getattr(usage, "total", 0) or 0))
            span.set_attribute("agent.finish_reason", str(getattr(ev, "finish_reason", "") or ""))
        elif etype == "run.failed":
            span.set_attribute("agent.error_type", str(getattr(ev, "error_type", "") or ""))
            _set_error(span, str(getattr(ev, "message", "") or "agent run failed"))
    except Exception:
        pass


def _set_error(span, message: str) -> None:
    try:
        from opentelemetry.trace import Status, StatusCode
        span.set_status(Status(StatusCode.ERROR, message))
    except Exception:
        pass


def _finish(span, buf: list) -> None:
    try:
        if buf:
            span.set_attribute(sc.OUTPUT_VALUE, "".join(buf))
            span.set_attribute(sc.OUTPUT_MIME_TYPE, sc.MIME_TEXT)
    except Exception:
        pass


def pai_agent_wrapper(func):
    @wraps(func)
    async def wrapper(self, ctx, *args, **kwargs):
        # Import lazily so the lean core (which imports this module only through a
        # guarded try/except) never hard-requires opentelemetry.
        try:
            from .tracer import get_tracer, is_enabled
            if not is_enabled():
                return await func(self, ctx, *args, **kwargs)
            tracer = get_tracer()
        except Exception:
            return await func(self, ctx, *args, **kwargs)

        async def traced():
            with tracer.start_as_current_span("agent.run") as span:
                _set_input(span, self, ctx)
                buf: list = []
                # Runs inside the span context, so run()'s body sees our span as
                # current and its @use_current_span keeps it current across the
                # generator's async iteration below.
                inner = await func(self, ctx, *args, **kwargs)
                try:
                    async for ev in inner:
                        _capture_event(span, ev, buf)
                        yield ev
                except Exception as e:  # pragma: no cover - defensive
                    span.record_exception(e)
                    _set_error(span, str(e))
                    raise
                finally:
                    _finish(span, buf)

        return traced()

    return wrapper
