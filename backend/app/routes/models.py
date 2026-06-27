from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from app.deps import AppState, get_state

router = APIRouter()


def _to_data(state: AppState):
    if state.router is not None:
        return [
            {"id": m.id, "object": "model", "created": 0, "owned_by": m.provider}
            for m in state.router.list_models()
        ]
    return [{"id": state.default_model, "object": "model", "created": 0, "owned_by": "openai"}]


@router.get("/v1/models")
async def list_models(state: AppState = Depends(get_state)):
    return JSONResponse({"object": "list", "data": _to_data(state)})


@router.post("/v1/models/reload")
async def reload_models(state: AppState = Depends(get_state)):
    if state.router is None:
        raise HTTPException(status_code=400, detail="no model router configured")
    state.router.reload_from_disk()
    return JSONResponse({"object": "list", "reloaded": True, "data": _to_data(state)})
