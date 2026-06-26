"""AgentEvent stream → OpenAI chat.completion(.chunk) JSON.

Entry points:
  serialize_chat_stream            – pure formatting, no side-effects (streaming)
  serialize_chat_stream_with_effects – streaming + output-guardrail + session-history save
                                       (parity with convert_gen_to_stream_chat_completions)
  serialize_chat_sync_with_effects – non-stream aggregation + output-guardrail + history save
                                       (parity with convert_gen_to_chat_completions)
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

    Ordering guarantee (mirrors utils.py):
      [...content/tool chunks] [safety_violation chunk?] [stop chunk with usage]

    The terminal stop chunk is always deferred to the epilogue so that:
      1. The safety chunk always precedes the stop chunk (even on RunCompleted).
      2. fail_fast paths (mid-stream guardrail reject, RunFailed) always emit
         exactly one stop chunk at the end, never zero.
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
    # tool_history_messages collects tool interactions for session history.
    # Note: session_history_manager does not persist tool_messages today, so
    # these are passed through as-is (placeholder arguments are intentionally inert).
    tool_history_messages: List[dict] = []
    check_tasks: List[asyncio.Task] = []
    output_check_result = TextCheckResult()
    fail_fast = False
    # pending_usage is set when RunCompleted fires; carried into the epilogue stop chunk.
    pending_usage: Optional[dict] = None

    # We use break (not return) for terminal events so that the code after the
    # try/finally block (guardrail epilogue) can still execute — a return inside
    # a try/finally would prevent any subsequent yields from reaching the caller.

    try:
        async for ev in events:
            # If a previous check already rejected, stop streaming
            if output_check_result.reject:
                logger.info(
                    "serialize_chat_stream_with_effects: output_check_result.reject=True, break"
                )
                fail_fast = True
                break

            # Terminal events: special handling (deferred stop chunk, usage capture)
            if isinstance(ev, RunFailed):
                # Include error message in final_content so history records it.
                # Emit the visible error content chunk now; the stop chunk is
                # deferred to the epilogue so ordering is always correct.
                final_content += ev.message
                yield _chunk(
                    chat_id,
                    model,
                    content=ev.message,
                    extra={"error_type": ev.error_type},
                )
                fail_fast = True
                break  # error path — skip guardrail checks, emit stop in epilogue

            elif isinstance(ev, RunCompleted):
                # Capture usage for the epilogue stop chunk; do NOT emit here.
                # The stop chunk is emitted after the guardrail safety chunk so
                # ordering is always: [content] [safety?] [stop+usage].
                pending_usage = {
                    "prompt_tokens": ev.usage.input,
                    "completion_tokens": ev.usage.output,
                    "total_tokens": ev.usage.total,
                }
                break  # normal completion — run guardrail epilogue below

            else:
                # Non-terminal events: accumulate side-effects then delegate
                # chunk formatting to the shared _event_to_chunks helper.
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

                # Delegate chunk formatting to the shared helper
                for s in _event_to_chunks(ev, chat_id, model):
                    yield s

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

    # --- Guardrail epilogue (runs after try/finally, after history is saved) ---
    # Final guardrail check on any leftover content not yet submitted
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

    # Always emit exactly one terminal stop chunk at the very end, on every path:
    #   - RunCompleted path: includes usage captured in pending_usage
    #   - RunFailed / fail_fast path: no usage (pending_usage is None)
    # This guarantees the safety chunk (if any) always precedes the stop chunk,
    # and every stream terminates with a stop chunk regardless of the exit path.
    yield _chunk(chat_id, model, finish_reason="stop", usage=pending_usage)


# ---------------------------------------------------------------------------
# Entry point 3: non-stream aggregation + guardrail + history save
# ---------------------------------------------------------------------------

async def serialize_chat_sync_with_effects(
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
) -> dict:
    """Consume the full AgentEvent stream once and return a single chat.completion dict.

    Side-effects (mirroring convert_gen_to_chat_completions in utils.py):
      (a) Final output-guardrail via checker.acheck_output over the assembled content
      (b) Session-history save via session_history_manager.save_messages (in finally)

    Output shape mirrors convert_gen_to_chat_completions:
      {"id","object":"chat.completion","created","model",
       "choices":[{"index":0,"message":{"role","content","reasoning_content"},
                   "finish_reason":"stop"}],
       "usage":{...}, "steps":[...], "citations":[], "citation_details":[],
       "safety_violation"?: True}

    NOTE on dropped/derived fields vs. the legacy ToolResultChunk-based path:
      - ``citations``/``citation_details`` were derived by JSON-parsing the raw tool
        result payload (extract_citations). AgentEvent.ToolResult only carries an
        opaque ``output`` string, so citation extraction is not reproduced here; both
        are returned as empty lists. This matches the streaming serializer, which also
        does not emit citations.
      - ``steps`` were full ToolResultChunk objects; here they are reconstructed from
        ToolStarted/ToolResult events (id/name/ok/output/error), enough for the
        frontend to render the tool-call trace.
    """
    from loguru import logger
    from extensions.guardrail.guardrail_check import TextCheckResult

    chat_id = "chatcmpl-" + uuid.uuid4().hex

    content = ""
    reasoning_content = ""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    actions: List[dict] = []
    observations: List[dict] = []
    steps: List[dict] = []
    citations: List[str] = []
    citation_details: List[dict] = []
    tool_history_messages: List[dict] = []
    safety_violation = False
    error_type: Optional[str] = None

    try:
        async for ev in events:
            if isinstance(ev, TextDelta):
                content += ev.text

            elif isinstance(ev, ReasoningDelta):
                reasoning_content += ev.text

            elif isinstance(ev, ToolStarted):
                actions.append({"id": ev.call_id, "name": ev.name})

            elif isinstance(ev, ToolResult):
                obs = {
                    "call_id": ev.call_id,
                    "ok": ev.ok,
                    "output": ev.output,
                    "error": ev.error,
                }
                observations.append(obs)
                steps.append({"name": ev.name, **obs})
                _collect_tool_history_from_event(ev, tool_history_messages)

            elif isinstance(ev, RunCompleted):
                usage = {
                    "prompt_tokens": ev.usage.input,
                    "completion_tokens": ev.usage.output,
                    "total_tokens": ev.usage.total,
                }

            elif isinstance(ev, RunFailed):
                # Make the error message visible in the final content and record
                # the error_type, mirroring the streaming path.
                content += ev.message
                error_type = ev.error_type

        # --- Final output-guardrail over the assembled content ---
        if enable_output_check and checker and content:
            current_result = TextCheckResult()
            await checker.acheck_output(text=content, current_result=current_result)
            if current_result.reject:
                logger.warning("serialize_chat_sync_with_effects: output check rejected")
                content = current_result.advice or guardrail_hint
                safety_violation = True

    finally:
        # Close session if provided (mirrors the streaming finally block)
        if session:
            try:
                await session.close()
                logger.info("serialize_chat_sync_with_effects: session closed.")
            except Exception:
                pass

        # Save session history
        if content and user_id and session_id and user_message:
            try:
                from service.cache.session_history_manager import session_history_manager

                assistant_message = {
                    "role": "assistant",
                    "content": content,
                }
                await session_history_manager.save_messages(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    tool_messages=tool_history_messages if tool_history_messages else None,
                )
                logger.info(
                    f"Session history saved (non-stream) for user={user_id}, session={session_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to save session history (non-stream): {e}", exc_info=True
                )

    result: dict = {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning_content if reasoning_content else None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
        "steps": steps,
        "citations": citations,
        "citation_details": citation_details,
    }
    if safety_violation:
        result["safety_violation"] = True
    if error_type:
        result["error_type"] = error_type
    return result
