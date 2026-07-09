from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.auth import require_user
from app.deps import AppState, get_state
from app.store.base import User
from app.conversations_view import group_conversation_messages

router = APIRouter()


class TruncatePayload(BaseModel):
    response_id: str


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


@router.get("/v1/conversations")
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    state: AppState = Depends(get_state),
    user: User = Depends(require_user),
):
    # Scope to the authenticated user — a client can no longer list another
    # user's conversations by passing their user_id.
    convs = await state.store.list_conversations(user_id=user.id, limit=limit, offset=offset)
    return JSONResponse(
        {
            "data": [
                {
                    "id": c.id,
                    "title": c.title,
                    "created_at": _iso(c.created_at),
                    "updated_at": _iso(c.updated_at),
                    "last_response_id": c.last_response_id,
                }
                for c in convs
            ]
        }
    )


@router.get("/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, state: AppState = Depends(get_state),
                           user: User = Depends(require_user)):
    conv = await state.store.get_conversation(conversation_id)
    if conv is None or conv.user_id != user.id:
        # 404 (not 403) so a probe can't distinguish "not yours" from "absent".
        raise HTTPException(status_code=404, detail="conversation not found")
    items = await state.store.get_conversation_items(conversation_id)
    responses = await state.store.list_conversation_responses(conversation_id)
    messages = group_conversation_messages(items, responses)
    return JSONResponse(
        {
            "id": conv.id,
            "title": conv.title,
            "created_at": _iso(conv.created_at),
            "updated_at": _iso(conv.updated_at),
            "latest_response_id": conv.last_response_id,
            "messages": messages,
        }
    )


@router.post("/v1/conversations/{conversation_id}/truncate")
async def truncate_conversation(conversation_id: str, payload: TruncatePayload,
                                state: AppState = Depends(get_state),
                                user: User = Depends(require_user)):
    """Remove the conversation's last turn so a regenerate re-runs the prompt in
    place instead of appending a duplicate. Enforces ownership and that the given
    response is actually the tail (409 otherwise)."""
    conv = await state.store.get_conversation(conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="conversation not found")
    try:
        prev = await state.store.truncate_last_turn(conversation_id, payload.response_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="conversation not found")
    except ValueError:
        raise HTTPException(status_code=409, detail="not the last response")
    return JSONResponse(
        {"conversation_id": conversation_id, "previous_response_id": prev}
    )


@router.delete("/v1/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, state: AppState = Depends(get_state),
                              user: User = Depends(require_user)):
    conv = await state.store.get_conversation(conversation_id)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="conversation not found")
    await state.store.delete_conversation(conversation_id)
    return JSONResponse(
        {"id": conversation_id, "object": "conversation.deleted", "deleted": True}
    )
