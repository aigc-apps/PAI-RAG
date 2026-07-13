from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, List, Optional

from loguru import logger

from agent.context import AgentContext, Attachment, RunVars
from agent.message import Message
from agent.budgeting import AgentMessageManager
from agent.message import ToolCall
from agent.tools.scope import ToolScope
from common.llm.models import ReasoningChunk, ErrorChunk
from agent.core.events import (
    RunStarted, TextDelta, ReasoningDelta, ToolStarted, ToolArgumentsDelta,
    ToolCompleted, ToolResult, RunCompleted, RunFailed, Usage,
)
from utils.constants import try_get_int_env
try:
    from extensions.trace.pai_agent_wrapper import pai_agent_wrapper
    from extensions.trace.base import use_current_span
    from opentelemetry import trace
except Exception:  # lean mode: no trace extension / opentelemetry
    def pai_agent_wrapper(func):           # passthrough decorator
        return func

    def use_current_span(_span):           # passthrough decorator factory
        def _deco(fn):
            return fn
        return _deco

    class _NoTrace:
        @staticmethod
        def get_current_span():
            return None

    trace = _NoTrace()


MAX_RECURSION_STEPS = try_get_int_env("MAX_RECURSION_STEPS", 20)
# 流式调用的"空闲超时":超过该秒数没有收到任何分片(package)即超时(非总时长)
LLM_STREAM_IDLE_TIMEOUT = try_get_int_env("LLM_STREAM_IDLE_TIMEOUT_SECONDS", 30)

# When a single assistant turn emits several independent tool calls, dispatch them
# concurrently (bounded) instead of one-at-a-time. This is how parallel fan-out —
# e.g. several spawn_subagent calls, or KB + code searches — actually runs in
# parallel, without a dedicated "batch" tool. Results are still stitched back into
# the message history and event stream in the model's original call order, so
# history stays replay-safe. Tuned modestly because each concurrent call may itself
# be an expensive subagent LLM run.
TOOL_DISPATCH_CONCURRENCY = try_get_int_env("TOOL_DISPATCH_CONCURRENCY", 8)

# Human-in-the-loop pause. When a tool result carries a notice with
# ``interrupt: true`` (e.g. the aliyun authorization card), the run stops after
# the current tool batch and hands control to the user instead of re-entering
# the LLM. This deterministic assistant line IS persisted (the notice/card is
# stream-only), so a reloaded conversation still explains the paused state.
_HITL_PAUSE_TEXT = (
    "我需要访问你的阿里云资源,但当前授权无法使用(尚未授权或凭证已失效)。请在上方的授权卡片"
    "或右上角头像菜单里完成阿里云授权,然后点击卡片上的「继续」按钮(或直接回复「继续」),我会接着执行。"
)


async def _iter_with_idle_timeout(stream, timeout: int):
    """Yield chunks, raising asyncio.TimeoutError if no chunk arrives within `timeout` seconds (idle/inter-chunk timeout, not total duration)."""
    it = stream.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(it.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return
        yield chunk


async def _dispatch_parallel(tools, pairs, scope):
    """Dispatch every tool call in ``pairs`` concurrently (bounded by
    TOOL_DISPATCH_CONCURRENCY) and return their results in the SAME order as
    ``pairs`` — so the caller can stitch them into history/events deterministically.

    Concurrency is safe because ToolBox.dispatch sets the ambient ToolScope on a
    contextvar, and asyncio.gather runs each coroutine in its own copied context,
    so per-call scope set/reset never races. dispatch also never raises (failures
    come back as ToolResult(ok=False)), so gather can't be torn down by one bad
    tool — no return_exceptions needed."""
    sem = asyncio.Semaphore(TOOL_DISPATCH_CONCURRENCY)

    async def _one(tc):
        async with sem:
            return await tools.dispatch(tc, scope=scope)

    return await asyncio.gather(*[_one(tc) for _, tc in pairs])


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
    blocks = []
    for a in attachments:
        # Replace any double-quote in the filename so it can't break the
        # name="..." attribute and confuse the model about block boundaries.
        safe_name = a.name.replace('"', "'")
        blocks.append(f'<attached_file name="{safe_name}">\n{a.body}\n</attached_file>')
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
        block = ctx.context_block
        if block:
            msgs.append(Message("system", block))
        msgs.append(render_current_turn(ctx.current_turn, ctx.attachments, ctx.hints, ctx.run_vars))
        logger.info("[agent] model input: {} msgs", len(msgs))
        return msgs

    async def _stream_turn(self, messages, tools, sink):
        """Stream one model turn. Emits AgentEvents (TextDelta/ReasoningDelta) live (no buffering).
        Accumulates usage into sink["usage"]; on an ErrorChunk or idle-timeout, sets sink["error"]
        to a RunFailed and stops. Results go through ``sink`` (a per-call dict) rather than
        instance state so an Agent has no cross-run mutable state."""
        wire = [m.to_wire() for m in messages]
        stream = await self.llm.astream(messages=wire, tools=tools.openai_schema() if tools else [])
        text, tool_calls = "", []
        started_tool_calls = sink.setdefault("started_tool_calls", set())
        argument_lengths = sink.setdefault("tool_argument_lengths", {})
        try:
            async for chunk in _iter_with_idle_timeout(stream, LLM_STREAM_IDLE_TIMEOUT):
                if isinstance(chunk, ErrorChunk):
                    sink["error"] = RunFailed(message=chunk.error_message or chunk.delta or "LLM error",
                                              error_type=chunk.error_type or "llm")
                    sink["last"] = (text, tool_calls)
                    return
                if chunk.tool_calls:
                    tool_calls = chunk.tool_calls
                    for raw in tool_calls:
                        if raw.type != "function" or raw.id is None or raw.function is None:
                            continue
                        name = raw.function.name or ""
                        if not name:
                            continue
                        if tools is None or not tools.get(name):
                            continue
                        if raw.id not in started_tool_calls:
                            started_tool_calls.add(raw.id)
                            yield ToolStarted(call_id=raw.id, name=name)
                        arguments = raw.function.arguments or ""
                        previous_len = argument_lengths.get(raw.id, 0)
                        if len(arguments) > previous_len:
                            argument_lengths[raw.id] = len(arguments)
                            yield ToolArgumentsDelta(
                                call_id=raw.id,
                                name=name,
                                delta=arguments[previous_len:],
                            )
                if chunk.usage:
                    sink["usage"] = Usage(input=chunk.usage.prompt_tokens or 0,
                                          output=chunk.usage.completion_tokens or 0,
                                          total=chunk.usage.total_tokens or 0)
                if isinstance(chunk, ReasoningChunk) and chunk.reasoning_delta:
                    yield ReasoningDelta(text=chunk.reasoning_delta)
                elif chunk.delta:
                    text += chunk.delta
                    yield TextDelta(text=chunk.delta)
        except asyncio.TimeoutError:
            logger.error(f"LLM stream idle >{LLM_STREAM_IDLE_TIMEOUT}s; aborting.")
            sink["error"] = RunFailed(
                message=f"模型调用超时：{LLM_STREAM_IDLE_TIMEOUT}s 内未收到任何响应分片。",
                error_type="llm_stream_timeout")
            sink["last"] = (text, tool_calls)
            return
        sink["last"] = (text, tool_calls)

    @pai_agent_wrapper
    async def run(self, ctx: AgentContext) -> AsyncIterator:
        """Execute the flat run loop. Decorated with @pai_agent_wrapper for tracing."""

        @use_current_span(trace.get_current_span())
        async def gen():
            messages = self.build_messages(ctx)

            # Per-run tool-body store: the budget writes full bodies here as it
            # offloads results from the window, and it rides the ToolScope so
            # read_handle can recover them in-run (before they're persisted). A
            # fresh dict per run keeps nested subagent runs isolated.
            run_bodies: dict = {}
            self.budget.run_bodies = run_bodies

            yield RunStarted(response_id=getattr(ctx, "response_id", "") or "resp_local")
            usage = Usage()
            for _step in range(self.max_steps):
                messages = self.budget.fit(messages)
                sink = {
                    "last": ("", []),
                    "error": None,
                    "usage": None,
                    "started_tool_calls": set(),
                    "tool_argument_lengths": {},
                }
                async for ev in self._stream_turn(messages, ctx.tools, sink):
                    yield ev
                if sink["usage"]:
                    usage = sink["usage"]
                if sink["error"] is not None:
                    yield sink["error"]
                    return
                text, raw_tcs = sink["last"]

                if not raw_tcs:
                    if text:
                        messages.append(Message("assistant", text))
                    yield RunCompleted(usage=usage, finish_reason="stop")
                    return

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

                for raw, tc in pairs:
                    if tc.id not in sink["started_tool_calls"]:
                        yield ToolStarted(call_id=tc.id, name=tc.name)
                    yield ToolCompleted(call_id=tc.id, name=tc.name, arguments=tc.arguments)

                scope = ToolScope(
                    user_id=ctx.user_id,
                    conversation_id=ctx.conversation_id,
                    metadata=ctx.metadata,
                    agent_id=ctx.agent_id,
                    skill_mounts=ctx.skill_mounts,
                    skill_fingerprint=ctx.skill_fingerprint,
                    run_bodies=run_bodies,
                )
                # Parallel fan-out: when the turn has several independent tool calls
                # and none returns directly (return_direct short-circuits the batch,
                # so it must stay sequential), dispatch them concurrently up front.
                # The loop below then consumes precomputed results in call order, so
                # message history and events keep the model's original ordering.
                # Single-call / return_direct turns fall through to the unchanged
                # await-per-call path (parallel_results stays None) — byte-identical.
                parallel_results = None
                if len(pairs) > 1 and not any(
                    ctx.tools.is_return_direct(tc.name) for _, tc in pairs
                ):
                    parallel_results = await _dispatch_parallel(ctx.tools, pairs, scope)

                interrupt_seen = False
                for idx, (raw, tc) in enumerate(pairs):
                    result = (
                        parallel_results[idx]
                        if parallel_results is not None
                        else await ctx.tools.dispatch(tc, scope=scope)
                    )
                    messages.append(Message("assistant", text if idx == 0 else None, tool_calls=[tc]))
                    capped = (self.budget.cap_tool_result(result.message.content)
                              if result.message.content else result.message.content)
                    messages.append(Message("tool", content=capped, tool_call_id=tc.id))
                    yield ToolResult(call_id=tc.id, name=tc.name, ok=result.ok,
                                     output=result.content, error=result.error,
                                     files=result.files, notice=result.notice)
                    if result.notice and result.notice.get("interrupt"):
                        interrupt_seen = True
                    if ctx.tools.is_return_direct(tc.name) and result.ok:
                        direct = _format_return_direct(result.content)
                        if direct:
                            yield TextDelta(text=direct)
                        yield RunCompleted(usage=usage, finish_reason="stop")
                        return

                # Human-in-the-loop: a tool asked to pause and yield to the user.
                # We finish the whole tool batch first (every function_call keeps
                # its paired function_call_output, so history stays replay-safe),
                # then stop instead of re-entering the LLM.
                if interrupt_seen:
                    yield TextDelta(text=_HITL_PAUSE_TEXT)
                    messages.append(Message("assistant", _HITL_PAUSE_TEXT))
                    yield RunCompleted(usage=usage, finish_reason="awaiting_user")
                    return
            yield RunCompleted(usage=usage, finish_reason="max_steps")

        return gen()
