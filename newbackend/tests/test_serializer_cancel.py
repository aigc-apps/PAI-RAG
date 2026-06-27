import sys, os, asyncio, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.core.events import RunStarted, TextDelta, RunCompleted, Usage
from api.protocol.responses_serializer import serialize_response_stream
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter

_ADAPTER = TypeAdapter(ResponseStreamEvent)


def _parse(lines):
    evs = []
    for ln in lines:
        for part in ln.splitlines():
            if part.startswith("data:"):
                payload = part[len("data:"):].strip()
                if payload and payload != "[DONE]":
                    evs.append(_ADAPTER.validate_python(json.loads(payload)))
    return evs


async def _slow_events(cancel: asyncio.Event):
    """Yield one text delta, then block forever — until cancel fires."""
    yield RunStarted(response_id="resp_c", conversation_id="conv_c")
    yield TextDelta(text="partial")
    await asyncio.Event().wait()  # never returns; the serializer must abandon us on cancel


async def _normal_events():
    yield RunStarted(response_id="resp_n", conversation_id="conv_n")
    yield TextDelta(text="hello")
    yield RunCompleted(usage=Usage(input=1, output=1, total=2))


def test_cancel_set_midstream_emits_incomplete_cancelled_terminal():
    async def run():
        cancel = asyncio.Event()
        sink = {}
        gen = serialize_response_stream(
            _slow_events(cancel), model="m", response_id="resp_c",
            conversation_id="conv_c", sink=sink, cancel=cancel,
        )
        out = []
        # consume the created/in_progress/text events, then cancel, then drain.
        agen = gen.__aiter__()
        # pull the first few events until we've seen the text delta
        async def pull_until_text():
            while True:
                chunk = await agen.__anext__()
                out.append(chunk)
                if "response.output_text.delta" in chunk:
                    return
        await pull_until_text()
        cancel.set()
        async for chunk in agen:
            out.append(chunk)
        evs = _parse(out)
        assert evs[-1].type == "response.incomplete"
        assert evs[-1].response.status == "cancelled"
        # partial text was preserved on the way out
        text = "".join(e.delta for e in evs if e.type == "response.output_text.delta")
        assert text == "partial"
        assert sink["response"]["status"] == "cancelled"
        assert any(it["type"] == "message" for it in sink["items"])
    asyncio.run(run())


def test_no_cancel_still_completes_normally():
    async def run():
        sink = {}
        out = [c async for c in serialize_response_stream(
            _normal_events(), model="m", response_id="resp_n",
            conversation_id="conv_n", sink=sink, cancel=asyncio.Event(),
        )]
        evs = _parse(out)
        assert evs[-1].type == "response.completed"
        assert sink["response"]["status"] == "completed"
    asyncio.run(run())
