from functools import wraps
import json
import os
from typing import cast, Awaitable
from common.llm.models import ChatResponseGenerator, ReasoningChunk, TextChunk
from opentelemetry.trace.status import Status, StatusCode
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
from extensions.trace import context as trace_context
from extensions.trace.utils import pydantic_to_dict
from loguru import logger
from extensions.trace.tracer import get_tracer


GEN_AI_MODEL = "gen_ai.request.model"
TEMPERATURE = "gen_ai.request.temperature"
MAX_TOKENS = "gen_ai.request.max_tokens"
INPUT_TOKENS = "gen_ai.usage.prompt_tokens"
OUTPUT_TOKENS = "gen_ai.usage.completion_tokens"
TOTAL_TOKENS = "gen_ai.usage.total_tokens"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"

INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_QUERY = "input.query"
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
GEN_AI_SPAN_KIND = "gen_ai.span.kind"
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
REASONING_CONTENT = "output.reasoning_content"

TOOL_NAME = "gen_ai.tool.name"
TOOL_DESCRIPTION = "gen_ai.tool.description"
TOOL_PARAMETERS = "gen_ai.tool.parameters"
STATUS_OK = Status(StatusCode.OK)



def pai_agent_wrapper(func):
    """decorator to capture input & output string of entry point (handle_chat in our case)."""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # if not enabled, directly return
        if os.getenv("TRACING_ENABLED", "false") != "true":
            return await func(self, *args, **kwargs)


        try:
            request_text = "[unknown]"
            state = kwargs.get("state")
            messages = state.messages if state else []
            # extract user input text
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

        async def wrapped_generator():
            with get_tracer().start_as_current_span(f"invoke_agent {self.__class__.__name__.lower()}") as span:
                span.set_attribute(GEN_AI_OPERATION_NAME, "invoke_agent")
                span.set_attribute(INPUT_MESSAGES, json.dumps(pydantic_to_dict(messages), ensure_ascii=False))

                span.set_attribute(INPUT_VALUE, request_text)
                span.set_attribute(GEN_AI_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
                for k, v in trace_context.get_context_vars():
                    if v:
                        span.set_attribute(k, v)

                final_output = ""
                final_reasoning_content = ""

                try:
                    response_gen = await func(self, *args, **kwargs)
                    response_gen = cast(ChatResponseGenerator, response_gen)

                    async for response in response_gen:
                        if isinstance(response, ReasoningChunk):
                            final_reasoning_content += response.reasoning_delta
                        elif isinstance(response, TextChunk):
                            final_output += response.delta

                        yield response

                    span.set_status(STATUS_OK)
                except Exception as stream_exc:
                    span.record_exception(stream_exc)
                    span.set_status(Status(StatusCode.ERROR, str(stream_exc)))
                    raise
                finally:
                    span.set_attribute(OUTPUT_VALUE, final_output)
                    if final_reasoning_content:
                        span.set_attribute(REASONING_CONTENT, final_reasoning_content)
        return wrapped_generator()


    return wrapper


async def instrument_async_call(
    async_fn: Awaitable,
    fn_args: dict,
) -> Awaitable:
    """
    Usage:
        return await instrument_async_call(async_fn, fn_args)
    """
    with get_tracer().start_as_current_span(f"execute_tool {async_fn.metadata.name}") as span:
        try:
            span.set_attribute(GEN_AI_SPAN_KIND, "TOOL")
            span.set_attribute(GEN_AI_OPERATION_NAME, "execute_tool")
            span.set_attribute(TOOL_NAME, async_fn.metadata.name)
            span.set_attribute(TOOL_DESCRIPTION, async_fn.metadata.description)
            span.set_attribute(TOOL_PARAMETERS, json.dumps(async_fn.metadata.fn_schema.model_json_schema(), ensure_ascii=False))
            span.set_attribute(INPUT_VALUE, json.dumps(fn_args, ensure_ascii=False))

            result = await async_fn.acall(**fn_args)
            if hasattr(result, "content") and result.content is not None:
                span.set_attribute(OUTPUT_VALUE, result.content)
            return result
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
