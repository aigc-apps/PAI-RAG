from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from app.auth import require_admin
from app.deps import AppState, get_state
from app.store.base import User

router = APIRouter()


def _to_data(state: AppState):
    if state.router is not None:
        return [
            {"id": m.qualified_id, "object": "model", "created": 0, "owned_by": m.provider}
            for m in state.router.list_models()
        ]
    return [{"id": state.default_model, "object": "model", "created": 0, "owned_by": "openai"}]


def _default_model_id(state: AppState) -> str:
    if state.router is not None:
        return state.router.default_model_id
    return state.default_model


@router.get("/v1/models")
async def list_models(state: AppState = Depends(get_state)):
    return JSONResponse({
        "object": "list",
        "default": _default_model_id(state),
        "data": _to_data(state),
    })


@router.post("/v1/models/reload")
async def reload_models(state: AppState = Depends(get_state),
                        admin: User = Depends(require_admin)):
    if state.router is None:
        raise HTTPException(status_code=400, detail="no model router configured")
    try:
        state.router.reload_from_disk()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return JSONResponse({
        "object": "list",
        "reloaded": True,
        "default": _default_model_id(state),
        "data": _to_data(state),
    })
