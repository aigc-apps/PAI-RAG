from __future__ import annotations
import traceback
from typing import List, Optional
from openai import AsyncOpenAI
from loguru import logger
from common.llm.models import (
    TextChunk,
    ReasoningChunk,
    ErrorChunk,
    update_tool_calls,
    THINK_START_TAG,
    THINK_END_TAG,
)

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 2


class LeanLLM:
    """Minimal streaming client over openai.AsyncOpenAI emitting the agent's chunk
    contract (TextChunk / ReasoningChunk / ErrorChunk). No model registry, no dashscope.

    Its only contract is what `Agent._stream_turn` consumes: `astream(messages, tools)`
    yielding chunks with `.delta` / `.tool_calls` / `.usage` (+ `.reasoning_delta` on
    ReasoningChunk). Tool-call deltas are coalesced by index via `update_tool_calls`.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        enable_thinking: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def astream(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs,
    ):
        tools_to_use = tools or None

        async def gen():
            tool_calls = []
            is_reasoning = True
            has_reasoning_content = False
            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=tools_to_use,
                    stream_options={"include_usage": True},
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": self.enable_thinking
                        },
                        "enable_thinking": self.enable_thinking,
                    },
                    **kwargs,
                )
                async for chunk in stream:
                    usage = getattr(chunk, "usage", None)
                    choices = getattr(chunk, "choices", None) or []
                    delta_obj = choices[0].delta if choices else None
                    if delta_obj is not None and getattr(
                        delta_obj, "tool_calls", None
                    ):
                        tool_calls = update_tool_calls(
                            tool_calls, delta_obj.tool_calls
                        )
                    content = (
                        getattr(delta_obj, "content", None) or ""
                        if delta_obj
                        else ""
                    )

                    if self.enable_thinking:
                        reasoning_delta = ""
                        reasoning_content = (
                            getattr(delta_obj, "reasoning_content", None)
                            if delta_obj
                            else None
                        )
                        if reasoning_content:
                            # Model emits a dedicated reasoning_content field.
                            has_reasoning_content = True
                            reasoning_delta = reasoning_content
                        elif content:
                            # Model inlines reasoning via <think>...</think>.
                            if has_reasoning_content:
                                is_reasoning = False
                            if is_reasoning:
                                end_pos = content.find(THINK_END_TAG)
                                if end_pos != -1:
                                    reasoning_delta = content[:end_pos]
                                    content = content[
                                        end_pos + len(THINK_END_TAG):
                                    ]
                                    is_reasoning = False
                                else:
                                    reasoning_delta = content.replace(
                                        THINK_START_TAG, ""
                                    )
                                    content = ""

                        if content or tool_calls:
                            yield TextChunk(
                                delta=content,
                                tool_calls=tool_calls,
                                usage=usage,
                            )
                        elif reasoning_delta:
                            yield ReasoningChunk(
                                delta=content,
                                reasoning_delta=reasoning_delta,
                                tool_calls=tool_calls,
                                usage=usage,
                            )
                        elif usage is not None:
                            yield TextChunk(
                                delta=content,
                                tool_calls=tool_calls,
                                usage=usage,
                            )
                    elif content or tool_calls or usage is not None:
                        yield TextChunk(
                            delta=content, tool_calls=tool_calls, usage=usage
                        )
            except Exception as ex:
                logger.error(f"LeanLLM stream error: {traceback.format_exc()}")
                yield ErrorChunk(
                    delta=f"{ex}",
                    error_message=str(ex),
                    exception=str(ex),
                    error_type="llm",
                )

        return gen()
