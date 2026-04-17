"""Server-Sent Events for file-status transitions.

Clients subscribe to `GET /v1/files/events?ids=file-a,file-b` and receive
`data: {...}\\n\\n` frames whenever a file reaches a terminal status
(`succeeded` or `failed`). The stream closes automatically once all
subscribed files are terminal.

Implementation: server-side polling of the DB every `POLL_INTERVAL_SECONDS`.
Simple, works under Redis Cluster, and avoids a pub/sub dependency. Can be
upgraded to Redis pub/sub later without changing the client contract.
"""
import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from db.db_context import create_db_session
from service.file.file_resource_service import FileResourceService
from service.injection import get_tenant_id


files_events_router = APIRouter()

POLL_INTERVAL_SECONDS = 1.0
STREAM_TIMEOUT_SECONDS = 300  # hard cap — matches the old sync-polling ceiling
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


async def _event_stream(
    file_ids: list[str], tenant_id: str
) -> AsyncIterator[bytes]:
    last_sent: dict[str, str] = {}
    deadline = asyncio.get_event_loop().time() + STREAM_TIMEOUT_SECONDS

    # Open-heartbeat comment so intermediaries don't close the connection.
    yield b": connected\n\n"

    while asyncio.get_event_loop().time() < deadline:
        async with create_db_session() as session:
            svc = FileResourceService(session)
            files = await svc.get_files(file_ids=file_ids, tenant_id=tenant_id)

        by_id = {f.id: f for f in files}
        all_terminal = True
        for fid in file_ids:
            entity = by_id.get(fid)
            if entity is None:
                # File was deleted or tenant mismatch — report once and move on.
                if last_sent.get(fid) != "not_found":
                    last_sent[fid] = "not_found"
                    yield _sse_frame({"id": fid, "status": "not_found"})
                continue
            if last_sent.get(fid) != entity.status:
                last_sent[fid] = entity.status
                yield _sse_frame({
                    "id": entity.id,
                    "status": entity.status,
                    "failed_reason": entity.failed_reason,
                })
            if entity.status not in TERMINAL_STATUSES:
                all_terminal = False

        if all_terminal:
            yield b"event: done\ndata: {}\n\n"
            return

        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    # Timed out — tell the client explicitly.
    yield b"event: timeout\ndata: {}\n\n"


def _sse_frame(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


@files_events_router.get("")
async def stream_file_events(
    ids: str = Query(..., description="Comma-separated file IDs to subscribe to"),
    tenant_id: str = Depends(get_tenant_id),
):
    file_ids = [fid.strip() for fid in ids.split(",") if fid.strip()]
    if not file_ids:
        # Client error surfaced as a one-shot SSE message so frontend handlers
        # don't have to switch on the content-type.
        async def _err() -> AsyncIterator[bytes]:
            yield b'event: error\ndata: {"message": "no ids"}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    return StreamingResponse(
        _event_stream(file_ids=file_ids, tenant_id=tenant_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
