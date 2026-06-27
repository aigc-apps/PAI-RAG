import sys, os, asyncio, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.core.events import (RunStarted, ToolStarted, ToolCompleted, ToolResult,
                               TextDelta, RunCompleted, Usage)
from api.protocol.responses_serializer import serialize_response_stream


async def _events():
    yield RunStarted(response_id="resp_1", conversation_id="conv_1")
    yield ToolStarted(call_id="c1", name="web_fetch")
    yield ToolCompleted(call_id="c1", name="web_fetch", arguments='{"url":"http://x"}')
    yield ToolResult(call_id="c1", name="web_fetch", ok=True, output="PAGE TEXT")
    yield TextDelta(text="done")
    yield RunCompleted(usage=Usage(input=1, output=1, total=2))


def _parse(chunks):
    evs = []
    for ln in chunks:
        for part in ln.splitlines():
            if part.startswith("data:"):
                p = part[len("data:"):].strip()
                if p and p != "[DONE]":
                    evs.append(json.loads(p))
    return evs


def test_stream_emits_tool_result_event():
    async def run():
        sink = {}
        chunks = [c async for c in serialize_response_stream(
            _events(), model="m", response_id="resp_1", conversation_id="conv_1", sink=sink)]
        evs = _parse(chunks)
        tr = [e for e in evs if e.get("type") == "response.tool_result"]
        assert len(tr) == 1
        assert tr[0]["call_id"] == "c1" and tr[0]["output"] == "PAGE TEXT" and tr[0]["ok"] is True
        assert isinstance(tr[0]["sequence_number"], int)
        # the function_call args.done event still precedes it
        types = [e["type"] for e in evs]
        assert "response.function_call_arguments.done" in types
        assert types.index("response.function_call_arguments.done") < types.index("response.tool_result")
    asyncio.run(run())
