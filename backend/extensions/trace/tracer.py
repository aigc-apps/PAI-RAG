from opentelemetry import trace, context


class ContextAwareTracer:
    def __init__(self, tracer):
        self._tracer = tracer

    def start_span(self, name, ctx=None, *args, **kwargs):
        if ctx is None:
            ctx = context.get_current()

        return self._tracer.start_span(name, ctx, *args, **kwargs)

    def start_as_current_span(self, *args, **kwargs):
        return self._tracer.start_as_current_span(*args, **kwargs)

def get_tracer():
    tracer = trace.get_tracer(__name__)
    return ContextAwareTracer(tracer)
