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


def _split_think(text: str, in_think: bool):
    """Split a content delta into ``(reasoning, answer, still_in_think)`` by
    scanning for ``<think>``/``</think>`` tags. Text counts as reasoning only
    while strictly inside a block, so a model that never emits the tags passes
    through untouched (all answer). ``in_think`` threads the open/closed state
    across chunks. Tags split across chunk boundaries are not recombined — the
    dedicated ``reasoning_content`` field is preferred when the model offers it."""
    reasoning, answer = "", ""
    while text:
        if in_think:
            end = text.find(THINK_END_TAG)
            if end == -1:
                reasoning += text
                text = ""
            else:
                reasoning += text[:end]
                text = text[end + len(THINK_END_TAG):]
                in_think = False
        else:
            start = text.find(THINK_START_TAG)
            if start == -1:
                answer += text
                text = ""
            else:
                answer += text[:start]
                text = text[start + len(THINK_START_TAG):]
                in_think = True
    return reasoning, answer, in_think


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
            in_think_block = False
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

                    # Split reasoning off UNCONDITIONALLY — never gate this on the
                    # request-side enable_thinking hint. A model can emit reasoning
                    # via a dedicated `reasoning_content` field (DeepSeek/GLM/Qwen)
                    # or inline as <think>...</think> whether or not we asked for
                    # it, and it must land in the reasoning channel, not the answer
                    # body. Detection triggers only on that field or real
                    # <think>/</think> tags, so a model that never thinks is
                    # unaffected.
                    reasoning_delta = ""
                    reasoning_content = (
                        getattr(delta_obj, "reasoning_content", None)
                        if delta_obj
                        else None
                    )
                    if reasoning_content:
                        has_reasoning_content = True
                        reasoning_delta = reasoning_content
                    elif content and not has_reasoning_content:
                        reasoning_delta, content, in_think_block = _split_think(
                            content, in_think_block
                        )

                    # Emit reasoning and answer as SEPARATE chunks: the agent routes
                    # a chunk to reasoning XOR text, so a single chunk carrying both
                    # a </think> boundary's reasoning tail and its answer head would
                    # otherwise drop the answer.
                    if reasoning_delta:
                        yield ReasoningChunk(
                            delta="",
                            reasoning_delta=reasoning_delta,
                            tool_calls=tool_calls,
                            usage=usage,
                        )
                        usage = None
                    if content or usage is not None or (
                        tool_calls and not reasoning_delta
                    ):
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
