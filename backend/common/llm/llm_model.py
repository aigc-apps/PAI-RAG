import traceback
from typing import List, Optional, cast
import uuid
from common.llm.models import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_RETRIES, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TIMEOUT, THINK_END_TAG, THINK_START_TAG, ChatResponseGenerator, ErrorChunk, ReasoningChunk, TextChunk
from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionToolParam
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from extensions.trace.base import use_current_span
from opentelemetry import trace


def update_tool_calls(
    tool_calls: List[ChoiceDeltaToolCall],
    tool_calls_delta: Optional[List[ChoiceDeltaToolCall]],
) -> List[ChoiceDeltaToolCall]:
    """
    Use the tool_calls_delta objects received from openai stream chunks
    to update the running tool_calls object.

    Handles parallel tool calls by matching on the ``index`` field.
    Each distinct index represents a separate tool call.

    Args:
        tool_calls: the accumulated list of tool calls so far.
        tool_calls_delta: new delta(s) from the current chunk.

    Returns:
        The updated tool calls list.
    """
    if tool_calls_delta is None or len(tool_calls_delta) == 0:
        return tool_calls

    for tc_delta in tool_calls_delta:
        # Find existing tool_call with the same index
        existing = None
        for tc in tool_calls:
            if tc.index == tc_delta.index:
                existing = tc
                break

        if existing is None:
            # First chunk for this index — start a new tool call entry
            tool_calls.append(tc_delta)
        else:
            # Continuation of an existing tool call — accumulate deltas
            assert existing.function is not None
            assert tc_delta.function is not None

            if existing.function.arguments is None:
                existing.function.arguments = ""
            if existing.function.name is None:
                existing.function.name = ""
            if existing.id is None:
                existing.id = ""

            existing.function.arguments += tc_delta.function.arguments or ""
            existing.function.name += tc_delta.function.name or ""
            # Only set id from delta if existing id is still empty;
            # avoids concatenating the same id across repeated chunks.
            if tc_delta.id and not existing.id:
                existing.id = tc_delta.id

    return tool_calls


class PaiLlm():
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        enable_thinking: bool = False,
        vision_support: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.vision_support = vision_support
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    async def astream(
        self,
        messages: List[dict],
        tools: List[ChatCompletionToolParam] = None,
        **kwargs,
    ) -> ChatResponseGenerator:
        @use_current_span(trace.get_current_span())
        async def gen():
            tool_calls: List[ChoiceDeltaToolCall] = []
            is_reasoning = True
            has_reasoning_content = False

            tools_to_use = tools
            if not tools_to_use:
                kwargs.pop("tool_choice", None) # pop choice if tools not given


            # in case we get duplicate tool call ids, like gemini tool_id is {index}_{tool_name}
            tool_tag = uuid.uuid4().hex[:6]
            try:
                response_gen = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=tools_to_use or None,
                    stream_options={"include_usage": True},
                    extra_body={"chat_template_kwargs":{"enable_thinking": self.enable_thinking}, "enable_thinking": self.enable_thinking},
                    **kwargs,
                )
                logger.info(f"Calling model {self.model}, enable_thinking: {self.enable_thinking}, temperature: {self.temperature}")
                async for chunk in response_gen:
                    chunk = cast(ChatCompletionChunk, chunk)

                    if not chunk.choices or not chunk.choices[0].delta:
                        if chunk.usage:
                            yield TextChunk(usage=chunk.usage)
                        continue


                    if chunk.choices[0].delta.tool_calls:
                        tool_calls = update_tool_calls(tool_calls, chunk.choices[0].delta.tool_calls)

                    for tool_call in tool_calls:
                        if not tool_call.id.startswith(tool_tag):
                            tool_call.id = tool_tag + tool_call.id

                    delta = chunk.choices[0].delta.content or ""
                    if self.enable_thinking:
                        reasoning_delta = ""
                        if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content:
                            has_reasoning_content = True
                            reasoning_delta = chunk.choices[0].delta.reasoning_content
                        elif delta:
                            if has_reasoning_content:
                                is_reasoning = False
                            if is_reasoning:
                                end_pos = delta.find(THINK_END_TAG)
                                if end_pos != -1:
                                    reasoning_delta = delta[:end_pos]
                                    delta = delta[end_pos + len(THINK_END_TAG):]
                                    is_reasoning = False
                                else:
                                    reasoning_delta = delta.replace(THINK_START_TAG, "")
                                    delta = ""

                        if delta or tool_calls:
                            yield TextChunk(
                                delta=delta,
                                tool_calls=tool_calls,
                                usage=chunk.usage,
                            )
                        elif reasoning_delta:
                            yield ReasoningChunk(
                                delta=delta,
                                reasoning_delta=reasoning_delta,
                                tool_calls=tool_calls,
                                usage=chunk.usage,
                            )
                    else:
                        yield TextChunk(
                            delta=delta,
                            tool_calls=tool_calls,
                            usage=chunk.usage,
                        )
            except Exception as ex:
                logger.error(f"Llm stream error: {traceback.format_exc()}")
                yield ErrorChunk(delta=f"{ex}", exception=str(ex), error_type="llm")

        return gen()


if __name__ == "__main__":
    import os
    import asyncio
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    model = "qwen-max"

    llm = PaiLlm(
        api_base=base_url,
        api_key=api_key,
        model=model,
        enable_thinking=False,
    )

    async def gen(messages: List[dict], tools=None):
        response_gen = llm.astream(
            messages=messages, tools=tools,
        )
        async for chunk in response_gen:
            if isinstance(chunk, ReasoningChunk):
                print("** " + chunk.reasoning_delta)
            elif chunk.delta:
                print("X " + chunk.delta)

            if chunk.tool_calls:
                print("DD ", chunk.tool_calls)

    print("="*12 + "Pure chat" + "=" * 12)
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    asyncio.run(gen(messages=messages))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather of a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            }
        }
    ]

    messages = [
        {"role": "user", "content": "今天杭州的天气怎么样?"}
    ]
    asyncio.run(gen(messages=messages, tools=tools))


    messages = [
        {"role": "user", "content": "今天星期几?"}
    ]
    asyncio.run(gen(messages=messages, tools=tools))
