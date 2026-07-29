import sys
import os
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))
from agent.core.events import TextDelta, RunCompleted, RunFailed, Usage, ToolResult, ToolStarted, ToolCompleted
from api.protocol.chat_serializer import (
    serialize_chat_stream,
    serialize_chat_stream_with_effects,
    serialize_chat_sync_with_effects,
)


async def _events(*evs):
    for e in evs:
        yield e


def _collect(gen):
    async def run():
        return [json.loads(s) async for s in gen]

    return asyncio.run(run())


def test_text_deltas_become_content_chunks():
    out = _collect(
        serialize_chat_stream(
            _events(
                TextDelta(text="he"),
                TextDelta(text="llo"),
                RunCompleted(usage=Usage(input=5, output=2, total=7)),
            ),
            model="m",
        )
    )
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in out)
    assert "hello" in content


def test_usage_reaches_final_chunk():  # regression: usage must not be dropped
    out = _collect(
        serialize_chat_stream(
            _events(
                TextDelta(text="hi"),
                RunCompleted(usage=Usage(input=5, output=9, total=14)),
            ),
            model="m",
        )
    )
    stop = [c for c in out if c["choices"][0].get("finish_reason") == "stop"][0]
    assert stop["usage"]["completion_tokens"] == 9 and stop["usage"]["total_tokens"] == 14


def test_run_failed_message_is_visible():  # regression: invisible-timeout bug
    out = _collect(
        serialize_chat_stream(
            _events(RunFailed(message="模型调用超时", error_type="llm_stream_timeout")),
            model="m",
        )
    )
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in out)
    assert "模型调用超时" in text


# ---------------------------------------------------------------------------
# Step 6: output-guardrail + history-save parity tests
# ---------------------------------------------------------------------------


def _collect_with_effects(gen):
    async def run():
        return [json.loads(s) async for s in gen]

    return asyncio.run(run())


def test_run_failed_triggers_history_save():
    """RunFailed must still save history — error text must appear in final_content."""
    mock_save = AsyncMock()
    mock_manager = MagicMock()
    mock_manager.save_messages = mock_save

    with patch(
        "api.protocol.chat_serializer.serialize_chat_stream_with_effects.__wrapped__"
        if hasattr(serialize_chat_stream_with_effects, "__wrapped__")
        else "service.cache.session_history_manager.session_history_manager",
        mock_manager,
    ):
        # Patch the import inside the function
        with patch.dict(
            "sys.modules",
            {
                "service.cache.session_history_manager": MagicMock(
                    session_history_manager=mock_manager
                )
            },
        ):
            out = _collect_with_effects(
                serialize_chat_stream_with_effects(
                    _events(RunFailed(message="模型调用超时", error_type="llm_stream_timeout")),
                    model="m",
                    user_id="u1",
                    session_id="s1",
                    user_message={"role": "user", "content": "hello"},
                )
            )

    # The error message must appear in streamed content
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in out)
    assert "模型调用超时" in text

    # history-save must have been called with the error text in the assistant message
    mock_save.assert_called_once()
    call_kwargs = mock_save.call_args.kwargs
    assert call_kwargs["user_id"] == "u1"
    assert call_kwargs["session_id"] == "s1"
    assert "模型调用超时" in call_kwargs["assistant_message"]["content"]


def test_rejecting_checker_yields_safety_violation_chunk():
    """A checker that rejects must emit a chunk with safety_violation=True and the advice."""
    from extensions.guardrail.guardrail_check import TextCheckResult

    async def fake_check_output(text, current_result):
        current_result.reject = True
        current_result.advice = "This content violates policy."

    fake_checker = MagicMock()
    fake_checker.acheck_output = fake_check_output

    # Use content long enough to trigger the chunked-check path
    long_text = "x" * 300  # > CHECK_OUTPUT_CHUNK_SIZE (200)

    with patch.dict(
        "sys.modules",
        {
            "service.cache.session_history_manager": MagicMock(
                session_history_manager=MagicMock(save_messages=AsyncMock())
            )
        },
    ):
        out = _collect_with_effects(
            serialize_chat_stream_with_effects(
                _events(
                    TextDelta(text=long_text),
                    RunCompleted(usage=Usage(input=1, output=1, total=2)),
                ),
                model="m",
                enable_output_check=True,
                checker=fake_checker,
                guardrail_hint="Default guardrail message",
            )
        )

    safety_chunks = [c for c in out if c.get("safety_violation")]
    assert safety_chunks, "Expected a safety_violation chunk"
    advice_text = safety_chunks[0]["choices"][0]["delta"].get("content", "")
    assert "violates policy" in advice_text


# ---------------------------------------------------------------------------
# Ordering + single-stop invariant tests
# ---------------------------------------------------------------------------


def test_with_effects_emits_single_stop_after_safety_on_reject():
    """On a guardrail rejection the safety chunk must precede the single stop chunk."""
    from extensions.guardrail.guardrail_check import TextCheckResult

    class _Checker:
        async def acheck_output(self, text, current_result):
            current_result.reject = True
            current_result.advice = "blocked"

    # Text long enough (> CHECK_OUTPUT_CHUNK_SIZE=200) to trigger the in-loop check.
    long_text = "hello world this is long enough to trigger a check " * 5  # ~255 chars

    with patch.dict(
        "sys.modules",
        {
            "service.cache.session_history_manager": MagicMock(
                session_history_manager=MagicMock(save_messages=AsyncMock())
            )
        },
    ):
        out = _collect_with_effects(
            serialize_chat_stream_with_effects(
                _events(
                    TextDelta(text=long_text),
                    RunCompleted(usage=Usage(input=1, output=2, total=3)),
                ),
                model="m",
                enable_output_check=True,
                checker=_Checker(),
                guardrail_hint="hint",
                user_id="u",
                session_id="s",
                user_message={"role": "user", "content": "q"},
            )
        )

    stops = [i for i, c in enumerate(out) if c["choices"][0].get("finish_reason") == "stop"]
    safety = [i for i, c in enumerate(out) if c.get("safety_violation")]

    assert len(stops) == 1, f"expected exactly one stop chunk, got {len(stops)}: {out}"
    assert safety, "expected a safety_violation chunk"
    assert safety[0] < stops[0], (
        f"safety chunk (index {safety[0]}) must precede stop chunk (index {stops[0]})"
    )


def test_with_effects_stop_carries_usage_when_completed():
    """The terminal stop chunk must include usage from RunCompleted."""
    with patch.dict(
        "sys.modules",
        {
            "service.cache.session_history_manager": MagicMock(
                session_history_manager=MagicMock(save_messages=AsyncMock())
            )
        },
    ):
        out = _collect_with_effects(
            serialize_chat_stream_with_effects(
                _events(
                    TextDelta(text="hi"),
                    RunCompleted(usage=Usage(input=5, output=9, total=14)),
                ),
                model="m",
                user_id="u",
                session_id="s",
                user_message={"role": "user", "content": "q"},
            )
        )

    stop_chunks = [c for c in out if c["choices"][0].get("finish_reason") == "stop"]
    assert len(stop_chunks) == 1, f"expected exactly one stop chunk, got {len(stop_chunks)}"
    stop = stop_chunks[0]
    assert stop["usage"]["completion_tokens"] == 9, f"expected completion_tokens=9, got {stop['usage']}"
    assert stop["usage"]["total_tokens"] == 14, f"expected total_tokens=14, got {stop['usage']}"


def test_with_effects_run_failed_emits_stop():
    """RunFailed path must always emit exactly one stop chunk."""
    with patch.dict(
        "sys.modules",
        {
            "service.cache.session_history_manager": MagicMock(
                session_history_manager=MagicMock(save_messages=AsyncMock())
            )
        },
    ):
        out = _collect_with_effects(
            serialize_chat_stream_with_effects(
                _events(RunFailed(message="timeout", error_type="llm_stream_timeout")),
                model="m",
                user_id="u",
                session_id="s",
                user_message={"role": "user", "content": "q"},
            )
        )

    stops = [c for c in out if c["choices"][0].get("finish_reason") == "stop"]
    assert len(stops) == 1, f"expected exactly one stop chunk on RunFailed, got {len(stops)}: {out}"


# ---------------------------------------------------------------------------
# Sync (non-stream) aggregator tests
# ---------------------------------------------------------------------------


def _run_sync(coro):
    return asyncio.run(coro)


def _mock_history_manager():
    """patch.dict context manager that mocks session_history_manager.save_messages."""
    return patch.dict(
        "sys.modules",
        {
            "service.cache.session_history_manager": MagicMock(
                session_history_manager=MagicMock(save_messages=AsyncMock())
            )
        },
    )


def test_sync_aggregates_content_and_usage():
    with _mock_history_manager():
        result = _run_sync(
            serialize_chat_sync_with_effects(
                _events(
                    TextDelta(text="he"),
                    TextDelta(text="llo"),
                    RunCompleted(usage=Usage(input=5, output=2, total=7)),
                ),
                model="m",
            )
        )

    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "hello"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["completion_tokens"] == 2
    assert result["usage"]["total_tokens"] == 7


def test_sync_run_failed_message_visible():
    with _mock_history_manager():
        result = _run_sync(
            serialize_chat_sync_with_effects(
                _events(
                    RunFailed(message="模型调用超时", error_type="llm_stream_timeout"),
                ),
                model="m",
            )
        )

    assert "模型调用超时" in result["choices"][0]["message"]["content"]
    assert result.get("error_type") == "llm_stream_timeout"


def test_sync_rejecting_checker_replaces_content_with_advice():
    async def fake_check_output(text, current_result):
        current_result.reject = True
        current_result.advice = "This content violates policy."

    fake_checker = MagicMock()
    fake_checker.acheck_output = fake_check_output

    with _mock_history_manager():
        result = _run_sync(
            serialize_chat_sync_with_effects(
                _events(
                    TextDelta(text="some disallowed content"),
                    RunCompleted(usage=Usage(input=1, output=1, total=2)),
                ),
                model="m",
                enable_output_check=True,
                checker=fake_checker,
                guardrail_hint="Default guardrail message",
            )
        )

    assert result["choices"][0]["message"]["content"] == "This content violates policy."
    assert result.get("safety_violation") is True


# ---------------------------------------------------------------------------
# Frontend shape regression tests (Fix 1 + Fix 2)
# ---------------------------------------------------------------------------


def test_tool_chunks_match_frontend_shape():
    """Stream serializer must emit actions/observation with the exact shapes the frontend parses.

    Frontend reads:
      - chunk.actions[].function?.name  (line 308 in usePaiChatThreadRuntime.tsx)
      - chunk.actions[].function?.arguments  (line 332 in usePaiChatThreadRuntime.tsx)
      - chunk.observation.tool.id  (line 366 in usePaiChatThreadRuntime.tsx)
      - chunk.observation.result  (line 369 in usePaiChatThreadRuntime.tsx)
      - chunk.observation.error   (line 370 in usePaiChatThreadRuntime.tsx)
    """
    out = _collect(
        serialize_chat_stream(
            _events(
                ToolStarted(call_id="c1", name="search"),
                ToolCompleted(call_id="c1", name="search", arguments='{"q":"x"}'),
                ToolResult(call_id="c1", name="search", ok=True, output="found"),
                RunCompleted(usage=Usage()),
            ),
            model="m",
        )
    )

    # actions chunk: must have function.name and function.arguments
    actions_chunks = [c["actions"] for c in out if c.get("actions")]
    assert actions_chunks, "Expected at least one chunk with actions"
    action = actions_chunks[0][0]
    assert action["function"]["name"] == "search", f"Got: {action}"
    assert action["function"]["arguments"] == '{"q":"x"}', f"Got: {action}"

    # observation chunk: must have tool.id, result, error
    obs_chunks = [c["observation"] for c in out if c.get("observation")]
    assert obs_chunks, "Expected at least one chunk with observation"
    obs = obs_chunks[0]
    assert obs["tool"]["id"] == "c1", f"Got: {obs}"
    assert obs["result"] == "found", f"Got: {obs}"
    assert obs["error"] is None, f"Got: {obs}"


def test_tool_actions_not_emitted_on_tool_started():
    """ToolStarted must NOT emit an actions chunk — arguments aren't available yet."""
    out = _collect(
        serialize_chat_stream(
            _events(
                ToolStarted(call_id="c1", name="search"),
                RunCompleted(usage=Usage()),
            ),
            model="m",
        )
    )
    actions_chunks = [c for c in out if c.get("actions")]
    assert not actions_chunks, f"Unexpected actions chunk on ToolStarted: {actions_chunks}"


def test_sync_tool_actions_observation_match_frontend_shape():
    """Sync serializer must also use the correct actions/observation shapes."""
    with _mock_history_manager():
        result = _run_sync(
            serialize_chat_sync_with_effects(
                _events(
                    ToolStarted(call_id="c1", name="search"),
                    ToolCompleted(call_id="c1", name="search", arguments='{"q":"x"}'),
                    ToolResult(call_id="c1", name="search", ok=True, output="found"),
                    RunCompleted(usage=Usage()),
                ),
                model="m",
            )
        )

    # actions shape
    assert result.get("actions"), f"Expected actions in sync result: {result}"
    action = result["actions"][0]
    assert action["function"]["name"] == "search", f"Got: {action}"
    assert action["function"]["arguments"] == '{"q":"x"}', f"Got: {action}"

    # observations shape
    assert result.get("observations"), f"Expected observations in sync result: {result}"
    obs = result["observations"][0]
    assert obs["tool"]["id"] == "c1", f"Got: {obs}"
    assert obs["result"] == "found", f"Got: {obs}"


def test_trace_id_emitted_on_stop_chunk():
    """trace_id must appear on the terminal stop chunk when a request_id is set."""
    import extensions.trace.context as ctx

    token = ctx._request_id_var.set("test-trace-123")
    try:
        out = _collect(
            serialize_chat_stream(
                _events(
                    TextDelta(text="hi"),
                    RunCompleted(usage=Usage(input=1, output=1, total=2)),
                ),
                model="m",
            )
        )
    finally:
        ctx._request_id_var.reset(token)

    stop_chunks = [c for c in out if c["choices"][0].get("finish_reason") == "stop"]
    assert stop_chunks, "Expected a stop chunk"
    assert stop_chunks[0].get("trace_id") == "test-trace-123", (
        f"Expected trace_id='test-trace-123' on stop chunk, got: {stop_chunks[0]}"
    )
