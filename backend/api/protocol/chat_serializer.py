"""AgentEvent stream → OpenAI chat.completion.chunk JSON strings.

Two entry points:
  serialize_chat_stream            – pure formatting, no side-effects
  serialize_chat_stream_with_effects – adds output-guardrail + session-history save
                                       (parity with convert_gen_to_stream_chat_completions)
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncIterator, List, Optional

from agent.core.events import (
    AgentEvent,
    ReasoningDelta,
    RunCompleted,
    RunFailed,
    TextDelta,
    ToolResult,
    ToolStarted,
)

MAX_TOOL_HISTORY_CHARS = 20_000
TOOL_HISTORY_TRUNCATED_MARKER = "\n...[content truncated]"


# ---------------------------------------------------------------------------
# Low-level chunk builder
# ---------------------------------------------------------------------------

def _chunk(
    chat_id: str,
    model: str,
    *,
    content: Optional[str] = None,
    reasoning: Optional[str] = None,
    finish_reason: Optional[str] = None,
    usage: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> str:
    delta: dict = {"role": "assistant"}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning

    body: dict = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        body["usage"] = usage
    if extra:
        body.update(extra)
    return json.dumps(body, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Per-event mapping (shared helper so both entry points stay DRY)
# ---------------------------------------------------------------------------

def _event_to_chunks(
    ev: AgentEvent, chat_id: str, model: str
) -> List[str]:
    """Return 0-N chunk JSON strings for a single AgentEvent."""
    if isinstance(ev, TextDelta):
        return [_chunk(chat_id, model, content=ev.text)]

    if isinstance(ev, ReasoningDelta):
        return [_chunk(chat_id, model, reasoning=ev.text)]

    if isinstance(ev, ToolStarted):
        return [
            _chunk(
                chat_id,
                model,
                content="",
                extra={"actions": [{"id": ev.call_id, "name": ev.name}]},
            )
        ]

    if isinstance(ev, ToolResult):
        return [
            _chunk(
                chat_id,
                model,
                content="",
                extra={
                    "observation": {
                        "call_id": ev.call_id,
                        "ok": ev.ok,
                        "output": ev.output,
                        "error": ev.error,
                    }
                },
            )
        ]

    if isinstance(ev, RunFailed):
        return [
            _chunk(
                chat_id,
                model,
                content=ev.message,
                extra={"error_type": ev.error_type},
            ),
            _chunk(chat_id, model, finish_reason="stop"),
        ]

    if isinstance(ev, RunCompleted):
        return [
            _chunk(
                chat_id,
                model,
                finish_reason="stop",
                usage={
                    "prompt_tokens": ev.usage.input,
                    "completion_tokens": ev.usage.output,
                    "total_tokens": ev.usage.total,
                },
            )
        ]

    # ToolCompleted: no chat.completion.chunk representation
    return []


# ---------------------------------------------------------------------------
# Entry point 1: pure formatting
# ---------------------------------------------------------------------------

async def serialize_chat_stream(
    events: AsyncIterator[AgentEvent],
    *,
    model: str,
) -> AsyncIterator[str]:
    """AgentEvent stream → chat.completion.chunk JSON strings (one per yield).

    The caller wraps each with the SSE 'data: ' prefix and trailing [DONE].
    """
    chat_id = "chatcmpl-" + uuid.uuid4().hex
    async for ev in events:
        for chunk_str in _event_to_chunks(ev, chat_id, model):
            yield chunk_str
        # Early-exit after terminal events
        if isinstance(ev, (RunCompleted, RunFailed)):
            return


# ---------------------------------------------------------------------------
# Tool-history collector (mirrors _collect_tool_history in utils.py)
# ---------------------------------------------------------------------------

def _collect_tool_history_from_event(
    ev: ToolResult, tool_history_messages: List[dict]
) -> None:
    """Append assistant-tool-call + tool-result messages for session history."""
    # ToolResult only carries call_id, name, output, error — no raw arguments
    # so we synthesise the assistant message with what we have.
    tool_history_messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": ev.call_id,
                    "type": "function",
                    "function": {
                        "name": ev.name,
                        "arguments": "{}",  # arguments not available in ToolResult
                    },
                }
            ],
        }
    )
    raw_content = ev.output or ev.error or ""
    if isinstance(raw_content, str) and len(raw_content) > MAX_TOOL_HISTORY_CHARS:
        raw_content = raw_content[:MAX_TOOL_HISTORY_CHARS] + TOOL_HISTORY_TRUNCATED_MARKER
    tool_history_messages.append(
        {
            "role": "tool",
            "tool_call_id": ev.call_id,
            "content": raw_content,
        }
    )


# ---------------------------------------------------------------------------
# Entry point 2: formatting + guardrail + history save
# ---------------------------------------------------------------------------

async def serialize_chat_stream_with_effects(
    events: AsyncIterator[AgentEvent],
    *,
    model: str,
    session=None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_message: Optional[dict] = None,
    enable_output_check: bool = False,
    checker=None,
    guardrail_hint: Optional[str] = None,
) -> AsyncIterator[str]:
    """AgentEvent stream → chat.completion.chunk JSON strings with side-effects.

    Side-effects (mirroring convert_gen_to_stream_chat_completions in utils.py):
      (a) Streaming output-guardrail via checker.acheck_output
      (b) Session-history save via session_history_manager.save_messages

    The caller wraps each yielded string with 'data: ' prefix and appends [DONE].
    """
    from loguru import logger
    from extensions.guardrail.config import CHECK_OUTPUT_CHUNK_OVERLAP, CHECK_OUTPUT_CHUNK_SIZE
    from extensions.guardrail.guardrail_check import TextCheckResult

    if enable_output_check and not checker:
        logger.warning(
            "serialize_chat_stream_with_effects: checker is None, disabling output check"
        )
        enable_output_check = False

    chat_id = "chatcmpl-" + uuid.uuid4().hex

    final_content = ""
    current_content = ""
    tool_history_messages: List[dict] = []
    check_tasks: List[asyncio.Task] = []
    output_check_result = TextCheckResult()
    fail_fast = False

    try:
        async for ev in events:
            # If a previous check already rejected, stop streaming
            if output_check_result.reject:
                logger.info(
                    "serialize_chat_stream_with_effects: output_check_result.reject=True, break"
                )
                fail_fast = True
                break

            # Accumulate text content for guardrail + history
            if isinstance(ev, TextDelta):
                final_content += ev.text
                current_content += ev.text
                if (
                    enable_output_check
                    and checker
                    and len(current_content) >= CHECK_OUTPUT_CHUNK_SIZE
                ):
                    check_tasks.append(
                        asyncio.create_task(
                            checker.acheck_output(
                                text=current_content,
                                current_result=output_check_result,
                            )
                        )
                    )
                    current_content = current_content[-CHECK_OUTPUT_CHUNK_OVERLAP:]

            elif isinstance(ev, ToolResult):
                _collect_tool_history_from_event(ev, tool_history_messages)

            elif isinstance(ev, RunFailed):
                # Include error message in final_content so history records it
                final_content += ev.message

            # Emit chunks
            for chunk_str in _event_to_chunks(ev, chat_id, model):
                yield chunk_str

            # Early-exit after terminal events
            if isinstance(ev, RunCompleted):
                return
            if isinstance(ev, RunFailed):
                fail_fast = True
                return

    finally:
        # Close session if provided (mirrors utils.py finally block)
        if session:
            try:
                await session.close()
                logger.info("serialize_chat_stream_with_effects: session closed.")
            except Exception:
                pass

        # Save session history
        if final_content and user_id and session_id and user_message:
            try:
                from service.cache.session_history_manager import session_history_manager

                assistant_message = {
                    "role": "assistant",
                    "content": final_content,
                }
                await session_history_manager.save_messages(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    tool_messages=tool_history_messages if tool_history_messages else None,
                )
                logger.info(
                    f"Session history saved for user={user_id}, session={session_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to save session history: {e}", exc_info=True
                )

    # Final guardrail check on leftover content
    if not fail_fast and len(current_content) > CHECK_OUTPUT_CHUNK_OVERLAP and enable_output_check and checker:
        check_tasks.append(
            asyncio.create_task(
                checker.acheck_output(
                    text=current_content, current_result=output_check_result
                )
            )
        )

    if not fail_fast and check_tasks:
        await asyncio.gather(*check_tasks)

    if output_check_result.reject:
        yield _chunk(
            chat_id,
            model,
            content=output_check_result.advice or guardrail_hint,
            extra={"safety_violation": True},
        )

    # Always emit a stop chunk (mirrors utils.py stop_chunk at end)
    yield _chunk(chat_id, model, finish_reason="stop")
