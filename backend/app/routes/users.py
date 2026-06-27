from __future__ import annotations
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from app.deps import AppState, get_state

router = APIRouter()


def _iso(dt):
    return dt.isoformat() if dt is not None else None


@router.get("/v1/users/{user_id}/memories")
async def list_user_memories(user_id: str, limit: int = 100, state: AppState = Depends(get_state)):
    mems = await state.store.list_memories(user_id, limit=limit)
    return JSONResponse({"data": [
        {"id": m.id, "text": m.text, "kind": m.kind,
         "created_at": _iso(m.created_at), "updated_at": _iso(m.updated_at)}
        for m in mems
    ]})


@router.delete("/v1/users/{user_id}/memories")
async def clear_user_memories(user_id: str, state: AppState = Depends(get_state)):
    await state.store.delete_user_memories(user_id)
    return JSONResponse({"deleted": True})


@router.delete("/v1/users/{user_id}/memories/{memory_id}")
async def delete_user_memory(user_id: str, memory_id: str, state: AppState = Depends(get_state)):
    await state.store.delete_memory(memory_id)
    return JSONResponse({"id": memory_id, "deleted": True})
