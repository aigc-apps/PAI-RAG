import asyncio
from contextlib import contextmanager
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Sequence,
    cast,
)

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
)
from openai.types.chat import (
    ChatCompletionChunk,
)
from pai_rag.app.api.models import (
    ChatCompletionRequest,
)
from llama_index.core.callbacks import CallbackManager

# dispatcher setup
from llama_index.core.instrumentation import get_dispatcher
from opentelemetry.context import attach, detach
from opentelemetry import trace

tracer = trace.get_tracer(__name__, tracer_provider=trace.get_tracer_provider())

dispatcher = get_dispatcher(__name__)


def llm_chat_callback() -> Callable:
    def wrap(f: Callable) -> Callable:
        @contextmanager
        def wrapper_logic(_self: Any) -> Generator[CallbackManager, None, None]:
            callback_manager = getattr(_self, "callback_manager", None)
            if not isinstance(callback_manager, CallbackManager):
                _self.callback_manager = CallbackManager()

            yield _self.callback_manager  # type: ignore

        async def wrapped_async_llm_chat(
            _self: Any, request: Any, **kwargs: Any
        ) -> Any:
            with wrapper_logic(_self) as callback_manager, callback_manager.as_trace(
                "chat"
            ):
                otel_span = tracer.start_span("llm.chat")
                context = trace.set_span_in_context(otel_span)
                token = attach(context)

                if isinstance(request, ChatCompletionRequest):
                    otel_span.set_attribute(
                        "input.value", request.messages[-1].blocks[0].text
                    )
                    otel_span.set_attribute("input.query", str(request))

                try:
                    f_return_val = await f(_self, request, **kwargs)
                except BaseException:
                    otel_span.end()
                    raise
                if isinstance(f_return_val, AsyncGenerator):
                    # intercept the generator and add a callback to the end
                    async def wrapped_gen() -> ChatResponseAsyncGen:
                        full_content = ""
                        try:
                            async for x in f_return_val:
                                try:
                                    chunk = ChatCompletionChunk.model_validate_json(
                                        x[6:]
                                    )
                                    full_content += chunk.choices[0].delta.content
                                except ValueError as e:
                                    print("Invalid JSON or data structure:", e)
                                yield cast(ChatResponse, x)
                                otel_span.set_attribute("output.value", full_content)
                        except BaseException:
                            otel_span.end()
                            raise
                        finally:
                            detach(token)
                        otel_span.end()

                    return wrapped_gen()
                else:
                    otel_span.end()

            return f_return_val

        def wrapped_llm_chat(
            _self: Any, messages: Sequence[ChatMessage], **kwargs: Any
        ) -> Any:
            with wrapper_logic(_self) as callback_manager, callback_manager.as_trace(
                "chat"
            ):
                try:
                    f_return_val = f(_self, messages, **kwargs)
                except BaseException:
                    raise
                if isinstance(f_return_val, Generator):
                    # intercept the generator and add a callback to the end
                    def wrapped_gen() -> ChatResponseGen:
                        full_content = ""
                        try:
                            for x in f_return_val:
                                yield cast(ChatResponse, x)
                                full_content += x.choices[0].delta.content
                        except BaseException:
                            raise

                    return wrapped_gen()
                else:
                    pass
            return f_return_val

        async def async_dummy_wrapper(_self: Any, *args: Any, **kwargs: Any) -> Any:
            return await f(_self, *args, **kwargs)

        def dummy_wrapper(_self: Any, *args: Any, **kwargs: Any) -> Any:
            return f(_self, *args, **kwargs)

        # check if already wrapped
        is_wrapped = getattr(f, "__wrapped__", False)
        if not is_wrapped:
            f.__wrapped__ = True  # type: ignore

        # Update the wrapper function to look like the wrapped function.
        # See e.g. https://github.com/python/cpython/blob/0abf997e75bd3a8b76d920d33cc64d5e6c2d380f/Lib/functools.py#L57
        for attr in (
            "__module__",
            "__name__",
            "__qualname__",
            "__doc__",
            "__annotations__",
            "__type_params__",
        ):
            if v := getattr(f, attr, None):
                setattr(async_dummy_wrapper, attr, v)
                setattr(wrapped_async_llm_chat, attr, v)
                setattr(dummy_wrapper, attr, v)
                setattr(wrapped_llm_chat, attr, v)

        if asyncio.iscoroutinefunction(f):
            if is_wrapped:
                return async_dummy_wrapper
            else:
                return wrapped_async_llm_chat
        else:
            if is_wrapped:
                return dummy_wrapper
            else:
                return wrapped_llm_chat

    return wrap
