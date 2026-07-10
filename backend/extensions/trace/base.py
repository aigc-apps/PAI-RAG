"""Context propagation helpers for async generators.

``Agent.run`` returns an async generator (``gen()``) whose body — the whole
step loop, including LLM streaming and tool dispatch — executes *lazily* as the
caller iterates it, detached from the stack frame that created it. Any span made
"current" at creation time is therefore NOT current inside the generator's
frames, so child spans (the auto-instrumented ``ChatCompletion``, the manual
``tool …`` spans) would attach to the wrong parent (or to no parent).

``use_current_span`` fixes that: it re-attaches the given span to the OTel
context for the duration of the generator's iteration, so every span opened
while the loop runs nests under it. A ``None`` span (lean mode / no active span)
degrades to a transparent passthrough.
"""
from __future__ import annotations

from functools import wraps

from opentelemetry import context as otel_context
from opentelemetry import trace


def use_current_span(span):
    """Decorator factory for an async-generator function.

    Wraps the generator so its iteration runs with ``span`` attached as the
    current span. Attaches once before the first ``__anext__`` and detaches when
    iteration ends (or the consumer stops early), so nesting holds across the
    ``await``-suspended points between yields.
    """
    def deco(fn):
        if span is None:
            return fn

        @wraps(fn)
        async def wrapper(*args, **kwargs):
            token = otel_context.attach(trace.set_span_in_context(span))
            try:
                async for item in fn(*args, **kwargs):
                    yield item
            finally:
                otel_context.detach(token)

        return wrapper

    return deco
