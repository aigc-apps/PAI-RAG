from functools import wraps
import os
import json
import time
from fastapi.responses import StreamingResponse
from opentelemetry import trace
from opentelemetry.context import attach, detach, Context
from opentelemetry.trace import set_span_in_context
from opentelemetry.trace.status import Status, StatusCode
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
from loguru import logger


tracer = trace.get_tracer(__name__, tracer_provider=trace.get_tracer_provider())


INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_QUERY = "input.query"
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
GEN_AI_SPAN_KIND = "gen_ai.span.kind"
CHAIN = OpenInferenceSpanKindValues.CHAIN.value

STATUS_OK = Status(StatusCode.OK)


def with_current_context(func):
    """decorator to pass context in streaming mode.

    To take current_context from out side functions and let `func` share the it.
    NOTE: `func` must provide a current_context parameter.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        current_context = kwargs.get("current_context") or next(
            (arg for arg in args if isinstance(arg, Context)), None
        )

        if current_context is None:
            current_span = trace.get_current_span()
            current_context = trace.set_span_in_context(current_span)

        token = attach(current_context)

        try:
            async for chunk in func(*args, **kwargs):
                yield chunk
        finally:
            detach(token)

    return wrapper


def _get_final_chunk_content(chunk: str):
    """return strip content if it's chunk from final call, empty otherwise"""
    # NOTE: this check should be align with generate_stream() in chat.py
    if chunk and chunk.startswith('0:"') and chunk.endswith('"\n'):
        return chunk[3:-2]

    # do not output if it's not chunks for final output
    return ""


def pai_agent_wrapper(func):
    """decorator to capture input & output string of entry point (handle_chat in our case)."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # if not enabled, directly return
        if os.getenv("TRACING_ENABLED", "false") != "true":
            return await func(*args, **kwargs)

        messages = kwargs.get("messages", [])
        try:
            for message in reversed(messages):
                if message.role == "user":
                    request_text = ""
                    for block in message.blocks:
                        if block.block_type == "text":
                            try:
                                j = json.loads(block.text)
                                request_text += j[0].get("text", "")
                            except Exception:
                                logger.warning(
                                    f"Failed to extract request text, block.text: {block.text}"
                                )
                                request_text += block.text
                    break
        except Exception as e:
            logger.warning(f"Failed to extract request text: {e}")
            request_text = "[unknown]"

        span = tracer.start_span(func.__qualname__)
        span.set_attribute(INPUT_VALUE, request_text)
        span.set_attribute(GEN_AI_SPAN_KIND, CHAIN)

        ctx = set_span_in_context(span)
        token = attach(ctx)

        try:
            response = await func(*args, **kwargs)

            if isinstance(response, StreamingResponse):
                original_body = response.body_iterator

                async def wrapped_generator():
                    final_output = ""
                    first_token_time = None
                    try:
                        async for chunk in original_body:
                            final_output += _get_final_chunk_content(chunk)
                            first_token_time = first_token_time or time.time_ns()
                            yield chunk
                        span.set_status(STATUS_OK)
                    except Exception as stream_exc:
                        span.record_exception(stream_exc)
                        span.set_status(Status(StatusCode.ERROR, str(stream_exc)))
                        raise
                    finally:
                        span.set_attribute(OUTPUT_VALUE, final_output)
                        span.end(end_time=first_token_time or time.time_ns())

                response.body_iterator = wrapped_generator()

            else:
                span.set_status(STATUS_OK)
                span.end()

            return response

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()
            raise

        finally:
            detach(token)

    return wrapper
