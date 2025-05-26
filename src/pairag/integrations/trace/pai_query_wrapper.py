import asyncio
from contextlib import contextmanager
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Generator,
)
import time
from loguru import logger

from openai.types.chat import ChatCompletionChunk
from pairag.chat.models import ChatCompletionRequest
from llama_index.core.callbacks import CallbackManager

from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
from opentelemetry.context import attach, detach
from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode

tracer = trace.get_tracer(__name__, tracer_provider=trace.get_tracer_provider())


INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_QUERY = "input.query"
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
GEN_AI_SPAN_KIND = "gen_ai.span.kind"
CHAIN = OpenInferenceSpanKindValues.CHAIN.value

STATUS_OK = Status(StatusCode.OK)
STATUS_ERROR = Status(StatusCode.ERROR)


def pai_query_wrapper() -> Callable:
    def wrap(f: Callable) -> Callable:
        @contextmanager
        def wrapper_logic(_self: Any) -> Generator[CallbackManager, None, None]:
            callback_manager = getattr(_self, "callback_manager", None)
            if not isinstance(callback_manager, CallbackManager):
                _self.callback_manager = CallbackManager()

            yield _self.callback_manager  # type: ignore

        async def wrapped_async_llm_chat(
            _self: Any, request: ChatCompletionRequest, **kwargs: Any
        ) -> Any:
            with wrapper_logic(_self) as callback_manager, callback_manager.as_trace(
                "chat"
            ):
                otel_span = tracer.start_span(f.__qualname__)
                end_time = 0
                context = trace.set_span_in_context(otel_span)
                token = attach(context)

                if request.messages:
                    # set latest input as input value
                    otel_span.set_attribute(
                        INPUT_VALUE, request.messages[-1].blocks[0].text
                    )
                otel_span.set_attribute(
                    INPUT_QUERY, request.model_dump_json(exclude_defaults=True)
                )
                otel_span.set_attribute(GEN_AI_SPAN_KIND, CHAIN)

                try:
                    f_return_val = await f(_self, request, **kwargs)
                except BaseException:
                    otel_span.set_status(STATUS_ERROR)
                    otel_span.end()
                    detach(token)
                    raise

                if isinstance(f_return_val, AsyncGenerator):
                    # make_completion_chunk_response
                    async def wrapped_gen():
                        full_content = ""
                        nonlocal end_time
                        try:
                            ctx = trace.set_span_in_context(otel_span)
                            t = attach(ctx)
                            async for x in f_return_val:
                                try:
                                    if x.startswith("data: "):
                                        chunk = ChatCompletionChunk.model_validate_json(
                                            x[6:]
                                        )
                                    else:
                                        chunk = ChatCompletionChunk.model_validate_json(
                                            x
                                        )
                                    full_content += chunk.choices[0].delta.content
                                    if not end_time:
                                        end_time = time.time_ns()
                                except ValueError as e:
                                    logger.error("Invalid JSON or data structure:", e)
                                yield x

                            otel_span.set_attribute(OUTPUT_VALUE, full_content)
                            # error response content, e.g., content_filer exception message
                            if full_content.startswith("Error code: "):
                                otel_span.set_status(STATUS_ERROR)
                            else:
                                otel_span.set_status(STATUS_OK)
                        except BaseException:
                            otel_span.set_status(STATUS_ERROR)
                            raise
                        finally:
                            otel_span.end(end_time=end_time or time.time_ns())
                            detach(t)

                    return wrapped_gen()
                else:
                    # make_completion_response
                    otel_span.set_attribute(
                        OUTPUT_VALUE, f_return_val.choices[0].message.content
                    )
                    otel_span.set_status(STATUS_OK)
                    otel_span.end(end_time=end_time or time.time_ns())
                    detach(token)

            return f_return_val

        def wrapped_llm_chat(
            _self: Any, request: ChatCompletionRequest, **kwargs: Any
        ) -> Any:
            with wrapper_logic(_self) as callback_manager, callback_manager.as_trace(
                "chat"
            ):
                otel_span = tracer.start_span(f.__qualname__)
                end_time = 0
                context = trace.set_span_in_context(otel_span)
                token = attach(context)

                try:
                    if request.messages:
                        otel_span.set_attribute(
                            INPUT_VALUE, request.messages[-1].blocks[0].text
                        )
                    otel_span.set_attribute(
                        INPUT_QUERY, request.model_dump_json(exclude_defaults=True)
                    )
                    otel_span.set_attribute(GEN_AI_SPAN_KIND, CHAIN)

                    f_return_val = f(_self, request, **kwargs)
                except BaseException:
                    otel_span.set_status(STATUS_ERROR)
                    otel_span.end()
                    detach(token)
                    raise

                if isinstance(f_return_val, Generator):
                    # make_completion_chunk_response
                    def wrapped_gen():
                        full_content = ""
                        nonlocal end_time
                        try:
                            ctx = trace.set_span_in_context(otel_span)
                            t = attach(ctx)
                            for x in f_return_val:
                                yield x
                                full_content += x.choices[0].delta.content
                                if not end_time:
                                    end_time = time.time_ns()

                            otel_span.set_attribute(OUTPUT_VALUE, full_content)
                            # error response content, e.g., content_filer exception message
                            if full_content.startswith("Error code: "):
                                otel_span.set_status(STATUS_ERROR)
                            else:
                                otel_span.set_status(STATUS_OK)
                        except BaseException:
                            otel_span.set_status(STATUS_ERROR)
                            raise
                        finally:
                            otel_span.end(end_time=end_time or time.time_ns())
                            detach(t)

                    return wrapped_gen()
                else:
                    # make_completion_response
                    otel_span.set_attribute(
                        OUTPUT_VALUE, f_return_val.choices[0].message.content
                    )

                otel_span.set_status(STATUS_OK)
                otel_span.end(end_time=end_time or time.time_ns())
                detach(token)
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
