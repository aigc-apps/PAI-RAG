from __future__ import annotations
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from app.schemas import ResponsesRequest
from app.builder import build_context
from app.deps import AppState, get_state
from app.store.base import Item, StoredResponse
from api.protocol.responses_serializer import (
    serialize_response_sync,
    serialize_response_stream,
)

router = APIRouter()


def _rid() -> str:
    return f"resp_{uuid.uuid4().hex}"


def _user_input_items(request: ResponsesRequest, response_id: str) -> list:
    """The user's turn, stored as a message item (history source of truth)."""
    if isinstance(request.input, str):
        text = request.input
    else:
        text = ""
        for it in request.input:
            if isinstance(it, dict) and isinstance(it.get("content"), str):
                text = it["content"]
    return [
        Item(
            type="message",
            role="user",
            content={"text": text},
            response_id=response_id,
        )
    ]


async def _persist(
    state: AppState,
    request: ResponsesRequest,
    response_id: str,
    conversation_id: str,
    store_items: list,
    status: str,
    usage: dict | None,
):
    items = _user_input_items(request, response_id)
    for d in store_items:
        items.append(
            Item(
                type=d["type"],
                role=d.get("role"),
                content=d["content"],
                response_id=response_id,
            )
        )
    await state.store.append_items(conversation_id, items)
    await state.store.save_response(
        StoredResponse(
            id=response_id,
            conversation_id=conversation_id,
            model=request.model or state.default_model,
            status=status,
            usage=usage,
            previous_response_id=request.previous_response_id,
        )
    )


@router.post("/v1/responses")
async def create_response(
    request: ResponsesRequest,
    req: Request,
    state: AppState = Depends(get_state),
):
    if not request.model:
        request.model = state.default_model
    try:
        # build_context is side-effect-free: conversation_id is ALWAYS non-None
        # (freshly minted for new turns, or the resolved existing one when linking).
        ctx, conversation_id = await build_context(request, state.store)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    response_id = _rid()
    agent = state.make_agent()
    events = await agent.run(ctx)

    if request.stream:

        async def gen():
            sink = {}
            async for chunk in serialize_response_stream(
                events,
                model=request.model,
                response_id=response_id,
                conversation_id=conversation_id,
                sink=sink,
            ):
                yield chunk
            if request.store and sink.get("response"):
                r = sink["response"]
                await _persist(
                    state,
                    request,
                    response_id,
                    conversation_id,
                    sink["items"],
                    r["status"],
                    r.get("usage"),
                )

        return StreamingResponse(gen(), media_type="text/event-stream")

    resp_dict, store_items = await serialize_response_sync(
        events,
        model=request.model,
        response_id=response_id,
        conversation_id=conversation_id,
    )
    if request.store:
        await _persist(
            state,
            request,
            response_id,
            conversation_id,
            store_items,
            resp_dict["status"],
            resp_dict.get("usage"),
        )
    return JSONResponse(resp_dict)


@router.get("/v1/responses/{response_id}")
async def get_response(response_id: str, state: AppState = Depends(get_state)):
    stored = await state.store.get_response(response_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="response not found")
    return JSONResponse(
        {
            "id": stored.id,
            "object": "response",
            "status": stored.status,
            "model": stored.model,
            "conversation": (
                {"id": stored.conversation_id}
                if stored.conversation_id
                else None
            ),
            "usage": stored.usage,
            "error": stored.error,
            "previous_response_id": stored.previous_response_id,
        }
    )


@router.delete("/v1/responses/{response_id}")
async def delete_response(
    response_id: str, state: AppState = Depends(get_state)
):
    stored = await state.store.get_response(response_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="response not found")
    await state.store.delete_response(response_id)
    return JSONResponse(
        {"id": response_id, "object": "response.deleted", "deleted": True}
    )
