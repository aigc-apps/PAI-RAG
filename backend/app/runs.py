from __future__ import annotations
import asyncio
import time
from typing import AsyncIterator, Awaitable, Callable, Dict, List, Optional
from loguru import logger
from api.protocol.responses_serializer import serialize_response_stream, make_failed_sse

RETENTION_SECONDS = 300

PersistFn = Callable[[Dict, str], Awaitable[None]]


class Run:
    """A single in-flight (or recently finished) streaming response.

    Holds the ordered SSE buffer (``events``); subscribers replay from a cursor
    then tail. ``sequence_number`` is monotonic from 1 with no gaps, so buffer
    index ``i`` holds the event with ``sequence_number == i+1`` — a resume cursor
    ``starting_after=N`` maps directly to buffer index ``N``.
    """

    def __init__(self, response_id: str, conversation_id: str, model: str,
                 user_id: Optional[str] = None):
        self.response_id = response_id
        self.conversation_id = conversation_id
        self.model = model
        self.user_id = user_id
        self.status = "in_progress"
        self.events: List[str] = []
        self.cancel = asyncio.Event()
        self.done = asyncio.Event()
        self.finished_at: Optional[float] = None
        self.task: Optional[asyncio.Task] = None
        self._cond = asyncio.Condition()

    async def append(self, chunk: str) -> None:
        async with self._cond:
            self.events.append(chunk)
            self._cond.notify_all()

    async def finish(self, status: str) -> None:
        self.status = status
        self.finished_at = time.monotonic()
        async with self._cond:
            self.done.set()
            self._cond.notify_all()


class RunManager:
    """In-memory, single-process registry of detached streaming runs."""

    def __init__(self, retention_seconds: int = RETENTION_SECONDS):
        self._runs: Dict[str, Run] = {}
        self._retention = retention_seconds

    def get(self, response_id: str) -> Optional[Run]:
        return self._runs.get(response_id)

    def _evict_expired(self) -> None:
        now = time.monotonic()
        stale = [
            rid for rid, r in self._runs.items()
            if r.finished_at is not None and now - r.finished_at > self._retention
        ]
        for rid in stale:
            self._runs.pop(rid, None)

    def start(self, *, events, model: str, response_id: str,
              conversation_id: str, persist: PersistFn,
              user_id: Optional[str] = None) -> Run:
        self._evict_expired()
        run = Run(response_id, conversation_id, model, user_id=user_id)
        self._runs[response_id] = run
        run.task = asyncio.create_task(self._pump(run, events, persist))
        return run

    async def _pump(self, run: Run, events, persist: PersistFn) -> None:
        sink: Dict = {}
        status = "failed"
        try:
            async for chunk in serialize_response_stream(
                events, model=run.model, response_id=run.response_id,
                conversation_id=run.conversation_id, sink=sink, cancel=run.cancel,
            ):
                await run.append(chunk)
            status = (sink.get("response") or {}).get("status", "completed")
        except Exception:  # noqa: BLE001 — a broken run must still finalize
            logger.exception(f"run {run.response_id} pump failed")
            status = "failed"
            await run.append(make_failed_sse(
                run.response_id, run.model, run.conversation_id,
                "stream interrupted",
            ))
            if not sink.get("response"):
                sink["response"] = {
                    "id": run.response_id,
                    "status": "failed",
                    "error": {"code": "server_error", "message": "stream interrupted"},
                    "usage": None,
                }
                sink["items"] = []
        finally:
            try:
                if sink.get("response"):
                    await persist(sink, status)
            except Exception:  # noqa: BLE001 — persistence failure must not hang the run
                logger.exception(f"run {run.response_id} persist failed")
            await run.finish(status)

    async def cancel(self, response_id: str) -> bool:
        run = self._runs.get(response_id)
        if run is None or run.done.is_set():
            return False
        run.cancel.set()
        return True

    async def subscribe(self, run: Run, starting_after: int = 0) -> AsyncIterator[str]:
        i = max(0, starting_after)
        while True:
            async with run._cond:
                while i >= len(run.events) and not run.done.is_set():
                    await run._cond.wait()
                new = run.events[i:]
                i = len(run.events)
                finished = run.done.is_set() and i >= len(run.events)
            for chunk in new:
                yield chunk
            if finished:
                break
