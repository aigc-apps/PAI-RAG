from functools import wraps
import json
import os
import time
from typing import cast
from common.llm.models import ChatResponseGenerator, ReasoningChunk, TextChunk
from opentelemetry.context import attach, detach
from opentelemetry.trace import set_span_in_context
from opentelemetry.trace.status import Status, StatusCode
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

from extensions.trace.utils import pydantic_to_dict
from extensions.trace import context as trace_context

from loguru import logger
from extensions.trace.tracer import get_tracer


TOOLS = "gen_ai.tools"
GEN_AI_MODEL = "gen_ai.request.model"
TEMPERATURE = "gen_ai.request.temperature"
MAX_TOKENS = "gen_ai.request.max_tokens"
INPUT_TOKENS = "gen_ai.usage.prompt_tokens"
OUTPUT_TOKENS = "gen_ai.usage.completion_tokens"
TOTAL_TOKENS = "gen_ai.usage.total_tokens"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"
TOOL_CALLS = "gen_ai.output.tool_calls"

INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_QUERY = "input.query"
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
GEN_AI_SPAN_KIND = "gen_ai.span.kind"
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
REASONING_CONTENT = "output.reasoning_content"

STATUS_OK = Status(StatusCode.OK)


def pai_llm_wrapper(func):
    """decorator to capture input & output string of entry point (handle_chat in our case)."""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # if not enabled, directly return
        if os.getenv("TRACING_ENABLED", "false") != "true":
            return await func(self, *args, **kwargs)

        try:
            request_text = "[unknown]"
            messages = kwargs.get("messages")
            for message in reversed(messages):
                if message["role"] == "user":
                    request_text = ""
                    if isinstance(message["content"], str):
                        request_text = message["content"]
                    else:
                        for message_part in message["content"]:
                            if message_part["type"] == "text":
                                request_text += message_part["text"]
                            elif message_part["type"] == "image":
                                request_text += message_part["image_url"]
                    break
        except Exception as e:
            logger.warning(f"Failed to extract request text: {e}")

        span = get_tracer().start_span(func.__qualname__)
        span.set_attribute(GEN_AI_MODEL, self.model)
        span.set_attribute(INPUT_MESSAGES, json.dumps(pydantic_to_dict(messages), ensure_ascii=False))

        tools = kwargs.get("tools")
        if tools:
            span.set_attribute(TOOLS, json.dumps(pydantic_to_dict(tools), ensure_ascii=False))

        span.set_attribute(TEMPERATURE, self.temperature)
        span.set_attribute(INPUT_VALUE, request_text)
        span.set_attribute(GEN_AI_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
        for k, v in trace_context.get_context_vars():
            if v:
                span.set_attribute(k, v)

        ctx = set_span_in_context(span)
        token = attach(ctx)

        try:
            response_gen = await func(self, *args, **kwargs)
            response_gen = cast(ChatResponseGenerator, response_gen)

            async def wrapped_generator():
                usage = None
                final_output = ""
                final_reasoning_content = ""
                tool_calls = []
                first_token_time = None
                try:
                    async for response in response_gen:
                        if response.usage:
                            usage = response.usage
                        if isinstance(response, ReasoningChunk):
                            final_reasoning_content += response.reasoning_delta
                        elif isinstance(response, TextChunk):
                            final_output += response.delta

                        if response.tool_calls:
                            tool_calls = response.tool_calls
                        first_token_time = first_token_time or time.time_ns()
                        yield response

                    span.set_status(STATUS_OK)
                except Exception as stream_exc:
                    span.record_exception(stream_exc)
                    span.set_status(Status(StatusCode.ERROR, str(stream_exc)))
                    raise
                finally:
                    if usage:
                        span.set_attribute(INPUT_TOKENS, usage.prompt_tokens)
                        span.set_attribute(OUTPUT_TOKENS, usage.completion_tokens)
                        span.set_attribute(TOTAL_TOKENS, usage.total_tokens)

                    raw_tool_calls = pydantic_to_dict(tool_calls)
                    output_message = {
                        "role": "assistant",
                        "content": final_output,
                        "reasoning_content": final_reasoning_content,
                        "tool_calls": raw_tool_calls
                    }

                    span.set_attribute(OUTPUT_MESSAGES, json.dumps([output_message], ensure_ascii=False))
                    if raw_tool_calls:
                        span.set_attribute(TOOL_CALLS, json.dumps(raw_tool_calls, ensure_ascii=False))

                    span.set_attribute(OUTPUT_VALUE, final_output)
                    if final_reasoning_content:
                        span.set_attribute(REASONING_CONTENT, final_reasoning_content)
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
