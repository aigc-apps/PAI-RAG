from functools import wraps
import os
import time
from typing import cast
from opentelemetry import trace
from opentelemetry.context import attach, detach
from opentelemetry.trace import set_span_in_context
from opentelemetry.trace.status import Status, StatusCode
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

from common.chat.models import ChatAgentRequest
from extensions.trace import context as trace_context

from loguru import logger

tracer = trace.get_tracer(__name__, tracer_provider=trace.get_tracer_provider())


INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_QUERY = "input.query"
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
GEN_AI_SPAN_KIND = "gen_ai.span.kind"
CHAIN = OpenInferenceSpanKindValues.CHAIN.value

STATUS_OK = Status(StatusCode.OK)


def pai_agent_wrapper(func):
    """decorator to capture input & output string of entry point (handle_chat in our case)."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # if not enabled, directly return
        if os.getenv("TRACING_ENABLED", "false") != "true":
            return await func(*args, **kwargs)

        try:
            chat_request = kwargs.get("chat_request")
            chat_request = cast(ChatAgentRequest, chat_request)
            for message in reversed(chat_request.messages):
                if message["role"] == "user":
                    request_text = ""
                    if isinstance(message["content"], str):
                        request_text = message["content"]
                    else:
                        for message_part in message["content"]:
                            if message_part["type"] == "text":
                                request_text += message_part["text"]
                    break
        except Exception as e:
            logger.warning(f"Failed to extract request text: {e}")
            request_text = "[unknown]"

        span = tracer.start_span(func.__qualname__)
        span.set_attribute(INPUT_VALUE, request_text)
        span.set_attribute(GEN_AI_SPAN_KIND, CHAIN)
        for k, v in trace_context.get_context_vars():
            if v:
                span.set_attribute(k, v)

        ctx = set_span_in_context(span)
        token = attach(ctx)

        try:
            response_gen = await func(*args, **kwargs)

            async def wrapped_generator():
                final_output = ""
                first_token_time = None
                try:
                    is_error = False
                    async for response in response_gen:
                        if response.message.role == "assistant":
                            final_output += response.delta
                        first_token_time = first_token_time or time.time_ns()
                        if response.message.additional_kwargs.get("failed"):
                            is_error = True
                        yield response
                    if not is_error:
                        span.set_status(STATUS_OK)
                    else:
                        span.set_status(Status(StatusCode.ERROR))
                except Exception as stream_exc:
                    span.record_exception(stream_exc)
                    span.set_status(Status(StatusCode.ERROR, str(stream_exc)))
                    raise
                finally:
                    span.set_attribute(OUTPUT_VALUE, final_output)
                    span.end(end_time=first_token_time or time.time_ns())

            return wrapped_generator()

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()
            raise

        finally:
            detach(token)

    return wrapper
