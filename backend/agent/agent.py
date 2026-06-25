from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, List, Optional

from loguru import logger

from agent.context import AgentContext, Attachment, RunVars
from agent.message import Message
from agent.budgeting import AgentMessageManager
from agent.tools import ToolBox
from agent.message import ToolCall
from common.llm.models import TextChunk, ReasoningChunk, ErrorChunk, ToolResultChunk
from utils.constants import try_get_int_env
from extensions.trace.pai_agent_wrapper import pai_agent_wrapper
from extensions.trace.base import use_current_span
from opentelemetry import trace


MAX_RECURSION_STEPS = try_get_int_env("MAX_RECURSION_STEPS", 20)
# 流式调用的"空闲超时":超过该秒数没有收到任何分片(package)即超时(非总时长)
LLM_STREAM_IDLE_TIMEOUT = try_get_int_env("LLM_STREAM_IDLE_TIMEOUT_SECONDS", 30)


async def _iter_with_idle_timeout(stream, timeout: int):
    it = stream.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(it.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return
        yield chunk


def _format_return_direct(content):
    """Port of tool_utils.check_and_handle_return_direct's success formatting."""
    if not content:
        return "Tool call successful, but no content returned."
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "result" in data:
            parts = [it.get("content", "") for it in data.get("result", []) if isinstance(it, dict)]
            joined = "\n\n".join(p for p in parts if p)
            return joined.strip() or content
    except Exception:
        pass
    return content


def _format_attachments(attachments: List[Attachment]) -> str:
    if not attachments:
        return ""
    blocks = [f'<attached_file name="{a.name}">\n{a.body}\n</attached_file>' for a in attachments]
    return "\n\n以下是用户本次上传的文件内容，请直接基于这些内容回答：\n\n" + "\n\n".join(blocks)


def render_current_turn(turn: Message, attachments: List[Attachment],
                        hints: List[str], run_vars: RunVars) -> Message:
    """Assemble the live user turn: time header + user text + attachment blocks + hints.
    The ONLY place these are combined. Handles both str and multimodal-list content."""
    prefix = f"[System Time: {run_vars.current_datetime}]\n"
    suffix = _format_attachments(attachments)
    if hints:
        suffix += "\n\n" + "\n\n".join(hints)

    if isinstance(turn.content, list):
        parts = [dict(p) for p in turn.content]
        for p in parts:
            if p.get("type") == "text":
                p["text"] = prefix + (p.get("text") or "") + suffix
                break
        else:
            parts.insert(0, {"type": "text", "text": prefix + suffix})
        return Message(role="user", content=parts)

    base = turn.content or ""
    return Message(role="user", content=prefix + base + suffix)


class Agent:
    def __init__(self, llm, max_steps: int = MAX_RECURSION_STEPS,
                 budget: Optional[AgentMessageManager] = None):
        self.llm = llm
        self.max_steps = max_steps
        self.budget = budget or AgentMessageManager(
            context_window=getattr(llm, "context_window", 0),
            max_output_tokens=getattr(llm, "max_tokens", 0),
        )

    @staticmethod
    def build_messages(ctx: AgentContext) -> List[Message]:
        msgs: List[Message] = [Message("system", ctx.system_prompt)]
        msgs += ctx.history
        msgs.append(render_current_turn(ctx.current_turn, ctx.attachments, ctx.hints, ctx.run_vars))
        logger.info(
            "[agent] model input: %d msgs; current turn head=%r",
            len(msgs),
            (msgs[-1].content if isinstance(msgs[-1].content, str) else "<multimodal>")[:200],
        )
        return msgs

    async def _stream_turn(self, messages, tools: ToolBox):
        """Stream one model turn. Yields TextChunk/ReasoningChunk live (no buffering).
        Stores (text, tool_calls) on self._last and any ErrorChunk on self._error."""
        wire = [m.to_wire() for m in messages]
        stream = await self.llm.astream(messages=wire, tools=tools.openai_schema() if tools else [])
        text, tool_calls = "", []
        async for chunk in _iter_with_idle_timeout(stream, LLM_STREAM_IDLE_TIMEOUT):
            if isinstance(chunk, ErrorChunk):
                self._error = chunk
                self._last = (text, tool_calls)
                return
            if chunk.tool_calls:
                tool_calls = chunk.tool_calls
            if isinstance(chunk, ReasoningChunk):
                yield chunk
            elif chunk.delta:
                text += chunk.delta
                yield TextChunk(delta=chunk.delta, usage=chunk.usage)
        self._last = (text, tool_calls)

    @pai_agent_wrapper
    async def run(self, ctx: AgentContext) -> AsyncIterator[TextChunk]:
        """Execute the flat run loop. Decorated with @pai_agent_wrapper for tracing."""

        @use_current_span(trace.get_current_span())
        async def gen():
            messages = self.build_messages(ctx)
            self._error = None

            for _step in range(self.max_steps):
                messages = self.budget.fit(messages)
                self._last = ("", [])

                async for ev in self._stream_turn(messages, ctx.tools):
                    yield ev

                if self._error is not None:
                    yield self._error
                    return

                text, raw_tcs = self._last

                if not raw_tcs:
                    if text:
                        messages.append(Message("assistant", text))
                    return  # plain text -> done

                pairs = [
                    (raw, ToolCall(
                        id=raw.id,
                        name=raw.function.name,
                        arguments=raw.function.arguments or "",
                    ))
                    for raw in raw_tcs
                    if raw.type == "function" and ctx.tools.get(raw.function.name)
                ]

                if not pairs:
                    bad = raw_tcs[0]
                    messages.append(Message("assistant", text or None,
                        tool_calls=[ToolCall(bad.id, bad.function.name, bad.function.arguments or "")]))
                    messages.append(Message("tool",
                        content=f"Error: tool '{bad.function.name}' not available.",
                        tool_call_id=bad.id))
                    continue

                for raw, _tc in pairs:
                    yield TextChunk(tool_calls=[raw])

                for idx, (raw, tc) in enumerate(pairs):
                    result = await ctx.tools.dispatch(tc)
                    messages.append(Message("assistant", text if idx == 0 else None, tool_calls=[tc]))
                    capped = (self.budget.cap_tool_result(result.message.content)
                              if result.message.content else result.message.content)
                    messages.append(Message("tool", content=capped, tool_call_id=tc.id))
                    yield ToolResultChunk(tool=raw, result=result.content, error=result.error)

                    if ctx.tools.is_return_direct(tc.name) and result.ok:
                        direct = _format_return_direct(result.content)
                        if direct:
                            yield TextChunk(delta=direct)
                        return

            yield TextChunk(delta=f"\n\nReached maximum iteration count ({self.max_steps}), task ended.")

        return gen()
