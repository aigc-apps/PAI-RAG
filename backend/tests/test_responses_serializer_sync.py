import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.core.events import (  # noqa: E402
    RunStarted,
    TextDelta,
    ReasoningDelta,
    ToolStarted,
    ToolCompleted,
    ToolResult,
    RunCompleted,
    RunFailed,
    Usage,
)
from api.protocol.responses_serializer import serialize_response_sync  # noqa: E402
from openai.types.responses import Response  # noqa: E402


async def _events(seq):
    for e in seq:
        yield e


def test_sync_text_response_parses_and_persists_items():
    async def run():
        events = _events(
            [
                RunStarted(response_id="resp_1", conversation_id="conv_1"),
                TextDelta(text="Hell"),
                TextDelta(text="o"),
                RunCompleted(
                    usage=Usage(input=3, output=5, total=8),
                    finish_reason="stop",
                ),
            ]
        )
        resp, items = await serialize_response_sync(
            events, model="m", response_id="resp_1", conversation_id="conv_1"
        )
        # Round-trips through the real OpenAI SDK type:
        parsed = Response.model_validate(resp)
        assert parsed.status == "completed"
        assert parsed.output[0].type == "message"
        assert parsed.output[0].content[0].text == "Hello"
        assert parsed.usage.total_tokens == 8
        # Store-ready items: assistant message persisted
        assert any(
            it["type"] == "message"
            and it["role"] == "assistant"
            and it["content"]["text"] == "Hello"
            for it in items
        )

    asyncio.run(run())


def test_sync_tool_call_response_emits_function_call_items():
    async def run():
        events = _events(
            [
                RunStarted(response_id="resp_2"),
                ToolStarted(call_id="c1", name="get"),
                ToolCompleted(call_id="c1", name="get", arguments='{"x":1}'),
                ToolResult(call_id="c1", name="get", ok=True, output="42"),
                TextDelta(text="done"),
                RunCompleted(usage=Usage(input=1, output=1, total=2)),
            ]
        )
        resp, items = await serialize_response_sync(
            events, model="m", response_id="resp_2", conversation_id=None
        )
        parsed = Response.model_validate(resp)
        types = [o.type for o in parsed.output]
        assert "function_call" in types and "message" in types
        fc = next(o for o in parsed.output if o.type == "function_call")
        assert (
            fc.name == "get"
            and fc.arguments == '{"x":1}'
            and fc.call_id == "c1"
        )
        # function_call + function_call_output persisted for history fidelity
        assert any(it["type"] == "function_call" for it in items)
        assert any(
            it["type"] == "function_call_output"
            and it["content"]["output"] == "42"
            for it in items
        )

    asyncio.run(run())


def test_sync_failure_sets_failed_status_and_error():
    async def run():
        events = _events(
            [
                RunStarted(response_id="resp_3"),
                RunFailed(message="kaboom", error_type="llm"),
            ]
        )
        resp, items = await serialize_response_sync(
            events, model="m", response_id="resp_3", conversation_id=None
        )
        parsed = Response.model_validate(resp)
        assert parsed.status == "failed"
        assert parsed.error is not None and "kaboom" in parsed.error.message

    asyncio.run(run())


def test_sync_reasoning_item_emitted_and_ordered_before_message():
    async def run():
        events = _events([
            RunStarted(response_id="resp_r"),
            ReasoningDelta(text="thinking..."),
            TextDelta(text="answer"),
            RunCompleted(usage=Usage(input=1, output=1, total=2)),
        ])
        resp, items = await serialize_response_sync(events, model="m", response_id="resp_r",
                                                    conversation_id=None)
        parsed = Response.model_validate(resp)
        # reasoning item comes before the assistant message, carried in the summary channel
        assert parsed.output[0].type == "reasoning"
        assert parsed.output[0].summary[0].text == "thinking..."
        assert parsed.output[0].summary[0].type == "summary_text"
        assert parsed.output[0].content == []
        assert parsed.output[-1].type == "message"
        assert parsed.output[-1].content[0].text == "answer"
        # reasoning persisted as a store item (skipped on replay, but recorded)
        assert any(it["type"] == "reasoning" and it["content"]["text"] == "thinking..." for it in items)
    asyncio.run(run())


def test_sync_tool_result_error_stores_error_text():
    async def run():
        events = _events([
            RunStarted(response_id="resp_e"),
            ToolStarted(call_id="c1", name="boom"),
            ToolCompleted(call_id="c1", name="boom", arguments="{}"),
            ToolResult(call_id="c1", name="boom", ok=False, output=None, error="exploded"),
            TextDelta(text="recovered"),
            RunCompleted(usage=Usage(input=1, output=1, total=2)),
        ])
        resp, items = await serialize_response_sync(events, model="m", response_id="resp_e",
                                                    conversation_id=None)
        out = next(it for it in items if it["type"] == "function_call_output")
        assert out["content"]["output"] == "exploded"
    asyncio.run(run())


def test_sync_persists_reasoning_content_and_tools_in_execution_order():
    async def run():
        events = _events([
            RunStarted(response_id="resp_timeline"),
            ReasoningDelta(text="先"),
            ReasoningDelta(text="分析"),
            TextDelta(text="准备调用"),
            ToolStarted(call_id="c1", name="get"),
            ToolCompleted(call_id="c1", name="get", arguments="{}"),
            ToolResult(call_id="c1", name="get", ok=True, output="42"),
            ReasoningDelta(text="检查"),
            ReasoningDelta(text="结果"),
            TextDelta(text="最终答案"),
            RunCompleted(usage=Usage(input=1, output=1, total=2)),
        ])
        _, items = await serialize_response_sync(
            events,
            model="m",
            response_id="resp_timeline",
            conversation_id=None,
        )
        message = next(
            item
            for item in items
            if item["type"] == "message" and item["role"] == "assistant"
        )
        assert message["content"]["timeline"] == [
            {"kind": "reasoning", "text": "先分析"},
            {"kind": "text", "text": "准备调用"},
            {"kind": "tool", "id": "c1"},
            {"kind": "reasoning", "text": "检查结果"},
            {"kind": "text", "text": "最终答案"},
        ]

    asyncio.run(run())


def test_sync_tool_completion_without_start_still_persists_one_timeline_step():
    async def run():
        events = _events([
            RunStarted(response_id="resp_tool_fallback"),
            ToolCompleted(call_id="c1", name="get", arguments="{}"),
            ToolCompleted(call_id="c1", name="get", arguments="{}"),
            RunCompleted(usage=Usage(input=1, output=1, total=2)),
        ])
        _, items = await serialize_response_sync(
            events,
            model="m",
            response_id="resp_tool_fallback",
            conversation_id=None,
        )
        message = next(item for item in items if item["type"] == "message")
        assert message["content"]["timeline"] == [
            {"kind": "tool", "id": "c1"}
        ]

    asyncio.run(run())
