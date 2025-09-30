from typing import List, Optional, cast
import uuid
from chat.llm.models import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_RETRIES, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TIMEOUT, THINK_END_TAG, THINK_START_TAG, ChatResponseGenerator, ReasoningChunk, TextChunk
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

    Args:
        tool_calls (List[ChoiceDeltaToolCall]): the list of tool calls
        tool_calls_delta (ChoiceDeltaToolCall): the delta to update tool_calls

    Returns:
        List[ChoiceDeltaToolCall]: the updated tool calls
    """
    # openai provides chunks consisting of tool_call deltas one tool at a time
    if tool_calls_delta is None or len(tool_calls_delta) == 0:
        return tool_calls

    tc_delta = tool_calls_delta[0]

    if len(tool_calls) == 0:
        tool_calls.append(tc_delta)
    else:
        # we need to either update latest tool_call or start a
        # new tool_call (i.e., multiple tools in this turn) and
        # accumulate that new tool_call with future delta chunks
        t = tool_calls[-1]
        if t.index != tc_delta.index:
            # the start of a new tool call, so append to our running tool_calls list
            tool_calls.append(tc_delta)
        else:
            # not the start of a new tool call, so update last item of tool_calls

            # validations to get passed by mypy
            assert t.function is not None
            assert tc_delta.function is not None

            # Initialize fields if they're None
            # OpenAI(or Compatible)'s streaming API can return partial tool call
            # information across multiple chunks where some fields may be None in
            # initial chunks and populated in subsequent ones
            if t.function.arguments is None:
                t.function.arguments = ""
            if t.function.name is None:
                t.function.name = ""
            if t.id is None:
                t.id = ""

            # Update with delta values
            t.function.arguments += tc_delta.function.arguments or ""
            t.function.name += tc_delta.function.name or ""
            t.id += tc_delta.id or ""
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

            response_gen = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tools or None,
                stream_options={"include_usage": True},
                extra_body={"chat_template_kwargs":{"enable_thinking": self.enable_thinking}},
                **kwargs,
            )

            # in case we get duplicate tool call ids, like gemini tool_id is {index}_{tool_name}
            tool_tag =  uuid.uuid4().hex[:6]
            async for chunk in response_gen:
                chunk = cast(ChatCompletionChunk, chunk)

                if not chunk.choices:
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
                    if hasattr(chunk.choices[0].delta, "reasoning_content"):
                        reasoning_delta = chunk.choices[0].delta.reasoning_content or ""
                    else:
                        if is_reasoning and delta:
                            end_pos = delta.find(THINK_END_TAG)
                            if end_pos != -1:
                                reasoning_delta = delta[:end_pos]
                                delta = delta[end_pos + len(THINK_END_TAG):]
                                is_reasoning = False
                            else:
                                reasoning_delta = delta.replace(THINK_START_TAG, "")
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
