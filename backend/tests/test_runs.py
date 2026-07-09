import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.runs import RunManager
from agent.core.events import RunStarted, TextDelta, RunCompleted, Usage


async def _events(texts):
    yield RunStarted(response_id="resp_1", conversation_id="conv_1")
    for t in texts:
        yield TextDelta(text=t)
    yield RunCompleted(usage=Usage(input=1, output=1, total=2))


async def _blocking_events():
    yield RunStarted(response_id="resp_1", conversation_id="conv_1")
    yield TextDelta(text="partial")
    await asyncio.Event().wait()


def test_run_persists_completed_with_no_subscriber():
    async def run():
        rm = RunManager()
        seen = {}

        async def persist(sink, status):
            seen["status"] = status
            seen["items"] = sink["items"]

        r = rm.start(events=_events(["he", "llo"]), model="m",
                     response_id="resp_1", conversation_id="conv_1", persist=persist)
        await asyncio.wait_for(r.done.wait(), timeout=2)
        assert r.status == "completed"
        assert seen["status"] == "completed"
        assert any(it["type"] == "message" and it["content"]["text"] == "hello"
                   for it in seen["items"])
    asyncio.run(run())


def test_subscribe_from_cursor_replays_then_tails():
    async def run():
        rm = RunManager()

        async def persist(sink, status):
            pass

        r = rm.start(events=_events(["a", "b", "c"]), model="m",
                     response_id="resp_1", conversation_id="conv_1", persist=persist)
        await asyncio.wait_for(r.done.wait(), timeout=2)
        # full replay from 0
        full = [c async for c in rm.subscribe(r, starting_after=0)]
        # resume from a cursor: only events with sequence_number > N (buffer index >= N)
        n = 3
        tail = [c async for c in rm.subscribe(r, starting_after=n)]
        assert len(full) > len(tail) and len(tail) == len(full) - n
        # the tail still ends with the terminal completed event
        assert "response.completed" in tail[-1]
    asyncio.run(run())


def test_live_subscriber_receives_streaming_events():
    async def run():
        rm = RunManager()

        async def persist(sink, status):
            pass

        r = rm.start(events=_events(["x", "y"]), model="m",
                     response_id="resp_1", conversation_id="conv_1", persist=persist)
        got = [c async for c in rm.subscribe(r, starting_after=0)]
        assert "response.created" in got[0]
        assert "response.completed" in got[-1]
    asyncio.run(run())


def test_cancel_finalizes_cancelled_and_persists_partial():
    async def run():
        rm = RunManager()
        seen = {}

        async def persist(sink, status):
            seen["status"] = status
            seen["items"] = sink["items"]

        r = rm.start(events=_blocking_events(), model="m",
                     response_id="resp_1", conversation_id="conv_1", persist=persist)
        # wait until the partial text has been buffered
        for _ in range(200):
            if any("response.output_text.delta" in c for c in r.events):
                break
            await asyncio.sleep(0.01)
        assert await rm.cancel("resp_1") is True
        await asyncio.wait_for(r.done.wait(), timeout=2)
        assert r.status == "cancelled"
        assert seen["status"] == "cancelled"
        assert any(it["content"].get("text") == "partial"
                   for it in seen["items"] if it["type"] == "message")
    asyncio.run(run())


def test_cancel_unknown_or_finished_returns_false():
    async def run():
        rm = RunManager()
        assert await rm.cancel("nope") is False
    asyncio.run(run())


def test_eviction_drops_finished_runs_after_ttl():
    async def run():
        rm = RunManager(retention_seconds=0)  # evict immediately on next start

        async def persist(sink, status):
            pass

        r1 = rm.start(events=_events(["a"]), model="m", response_id="resp_1",
                      conversation_id="conv_1", persist=persist)
        await asyncio.wait_for(r1.done.wait(), timeout=2)
        # next start sweeps expired finished runs
        r2 = rm.start(events=_events(["b"]), model="m", response_id="resp_2",
                      conversation_id="conv_1", persist=persist)
        await asyncio.wait_for(r2.done.wait(), timeout=2)
        assert rm.get("resp_1") is None
        assert rm.get("resp_2") is not None
    asyncio.run(run())
