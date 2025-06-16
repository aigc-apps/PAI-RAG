from functools import wraps
import os
import json
import time
from fastapi.responses import StreamingResponse
from opentelemetry import trace
from opentelemetry.context import attach, detach
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
                        is_error = False
                        async for chunk in original_body:
                            final_output += _get_final_chunk_content(chunk)
                            first_token_time = first_token_time or time.time_ns()
                            if '"finishReason":"error"' in chunk:
                                is_error = True
                            yield chunk
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
