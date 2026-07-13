import asyncio
import contextvars

from extensions.trace import base


def test_span_context_is_attached_and_detached_within_each_iteration(monkeypatch):
    async def scenario():
        current = contextvars.ContextVar("test_trace_context", default=None)

        class FakeContext:
            @staticmethod
            def attach(value):
                return current.set(value)

            @staticmethod
            def detach(token):
                current.reset(token)

        class FakeTrace:
            @staticmethod
            def set_span_in_context(span):
                return span

        monkeypatch.setattr(base, "otel_context", FakeContext)
        monkeypatch.setattr(base, "trace", FakeTrace)

        @base.use_current_span("span")
        async def source():
            yield current.get()
            yield current.get()

        iterator = source().__aiter__()
        first = await asyncio.create_task(iterator.__anext__())
        second = await asyncio.create_task(iterator.__anext__())

        assert first == second == "span"

    asyncio.run(scenario())
