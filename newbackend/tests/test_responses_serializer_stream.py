# tests/app/test_responses_serializer_stream.py
import sys, os, asyncio, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.core.events import (
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
from api.protocol.responses_serializer import serialize_response_stream
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter

_ADAPTER = TypeAdapter(ResponseStreamEvent)


def _parse(lines):
    evs = []
    for ln in lines:
        for part in ln.splitlines():
            if part.startswith("data:"):
                payload = part[len("data:") :].strip()
                if payload and payload != "[DONE]":
                    evs.append(_ADAPTER.validate_python(json.loads(payload)))
    return evs


async def _events(seq):
    for e in seq:
        yield e


def test_stream_text_event_order_and_sink():
    async def run():
        sink = {}
        gen = serialize_response_stream(
            _events(
                [
                    RunStarted(response_id="resp_1", conversation_id="conv_1"),
                    TextDelta(text="Hi"),
                    TextDelta(text="!"),
                    RunCompleted(usage=Usage(input=1, output=1, total=2)),
                ]
            ),
            model="m",
            response_id="resp_1",
            conversation_id="conv_1",
            sink=sink,
        )
        lines = [chunk async for chunk in gen]
        evs = _parse(lines)
        types = [e.type for e in evs]
        assert types[0] == "response.created"
        assert "response.in_progress" in types
        assert "response.output_text.delta" in types
        assert types[-1] == "response.completed"
        # full text reconstructable from deltas
        text = "".join(
            e.delta for e in evs if e.type == "response.output_text.delta"
        )
        assert text == "Hi!"
        # sink carries the final response + persisted items
        assert sink["response"]["status"] == "completed"
        assert any(it["type"] == "message" for it in sink["items"])

    asyncio.run(run())


def test_stream_tool_call_events():
    async def run():
        sink = {}
        gen = serialize_response_stream(
            _events(
                [
                    RunStarted(response_id="resp_2"),
                    ToolStarted(call_id="c1", name="get"),
                    ToolCompleted(
                        call_id="c1", name="get", arguments='{"x":1}'
                    ),
                    ToolResult(call_id="c1", name="get", ok=True, output="42"),
                    TextDelta(text="ok"),
                    RunCompleted(usage=Usage(input=1, output=1, total=2)),
                ]
            ),
            model="m",
            response_id="resp_2",
            conversation_id=None,
            sink=sink,
        )
        evs = _parse([c async for c in gen])
        types = [e.type for e in evs]
        assert "response.function_call_arguments.done" in types
        done = next(
            e for e in evs if e.type == "response.function_call_arguments.done"
        )
        assert done.arguments == '{"x":1}' and done.name == "get"

    asyncio.run(run())


def test_stream_output_indices_are_distinct_for_each_item():
    async def run():
        sink = {}
        gen = serialize_response_stream(
            _events([RunStarted(response_id="resp_4"),
                     ToolStarted(call_id="c1", name="get"),
                     ToolCompleted(call_id="c1", name="get", arguments="{}"),
                     ToolResult(call_id="c1", name="get", ok=True, output="1"),
                     TextDelta(text="answer"),
                     RunCompleted(usage=Usage(input=1, output=1, total=2))]),
            model="m", response_id="resp_4", conversation_id=None, sink=sink)
        evs = _parse([c async for c in gen])
        added = [e for e in evs if e.type == "response.output_item.added"]
        idxs = [e.output_index for e in added]
        # each opened output item must have a distinct output_index
        assert len(idxs) == len(set(idxs)), f"colliding output_index: {idxs}"
        # at least the function_call and the message item were opened
        assert len(added) >= 2
    asyncio.run(run())


def test_stream_reasoning_summary_envelope_before_message():
    async def run():
        sink = {}
        gen = serialize_response_stream(
            _events(
                [
                    RunStarted(response_id="resp_r"),
                    ReasoningDelta(text="think"),
                    ReasoningDelta(text="ing"),
                    TextDelta(text="answer"),
                    RunCompleted(usage=Usage(input=1, output=1, total=2)),
                ]
            ),
            model="m",
            response_id="resp_r",
            conversation_id=None,
            sink=sink,
        )
        evs = _parse([c async for c in gen])
        types = [e.type for e in evs]

        # the reasoning-summary sub-events appear in order
        part_added = types.index("response.reasoning_summary_part.added")
        text_done = types.index("response.reasoning_summary_text.done")
        part_done = types.index("response.reasoning_summary_part.done")
        delta_idxs = [
            i
            for i, t in enumerate(types)
            if t == "response.reasoning_summary_text.delta"
        ]
        assert len(delta_idxs) == 2
        assert part_added < delta_idxs[0] < delta_idxs[1] < text_done < part_done

        # the reasoning output item opens before and closes after the summary events
        reasoning_added = next(
            i
            for i, e in enumerate(evs)
            if e.type == "response.output_item.added"
            and e.item.type == "reasoning"
        )
        reasoning_done = next(
            i
            for i, e in enumerate(evs)
            if e.type == "response.output_item.done"
            and e.item.type == "reasoning"
        )
        assert reasoning_added < part_added
        assert part_done < reasoning_done

        # reconstructed summary text from deltas
        summary_text = "".join(
            e.delta
            for e in evs
            if e.type == "response.reasoning_summary_text.delta"
        )
        assert summary_text == "thinking"

        # the reasoning item's done event carries the full summary
        reasoning_done_ev = evs[reasoning_done]
        assert reasoning_done_ev.item.summary[0].text == "thinking"
        assert reasoning_done_ev.item.summary[0].type == "summary_text"

        # reasoning item comes before the MESSAGE output item
        msg_added = next(
            i
            for i, e in enumerate(evs)
            if e.type == "response.output_item.added"
            and e.item.type == "message"
        )
        assert reasoning_added < msg_added
        assert reasoning_done < msg_added

        # final response output has reasoning first with the summary populated
        assert sink["response"]["output"][0]["type"] == "reasoning"
        assert (
            sink["response"]["output"][0]["summary"][0]["text"] == "thinking"
        )

    asyncio.run(run())


def test_stream_failure_emits_failed_event():
    async def run():
        sink = {}
        gen = serialize_response_stream(
            _events(
                [
                    RunStarted(response_id="resp_3"),
                    RunFailed(message="boom", error_type="llm"),
                ]
            ),
            model="m",
            response_id="resp_3",
            conversation_id=None,
            sink=sink,
        )
        evs = _parse([c async for c in gen])
        assert evs[-1].type == "response.failed"
        assert sink["response"]["status"] == "failed"

    asyncio.run(run())
