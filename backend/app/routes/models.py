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
            {
                "id": m.qualified_id,
                "object": "model",
                "created": 0,
                "owned_by": m.provider,
                "type": m.type,
                "dimension": m.dimension,
            }
            for m in state.router.list_models()
        ]
    return [{"id": state.default_model, "object": "model", "created": 0,
             "owned_by": "openai", "type": "chat", "dimension": None}]


def _default_model_id(state: AppState) -> str:
    if state.router is not None:
        return state.router.default_model_id
    return state.default_model


def _envelope(state: AppState, **extra) -> dict:
    router = state.router
    return {
        "object": "list",
        "default": _default_model_id(state),
        "default_embedding": router.default_embedding_model_id if router else None,
        "default_rerank": router.default_rerank_model_id if router else None,
        "data": _to_data(state),
        **extra,
    }


@router.get("/v1/models")
async def list_models(state: AppState = Depends(get_state)):
    return JSONResponse(_envelope(state))


@router.post("/v1/models/reload")
async def reload_models(state: AppState = Depends(get_state),
                        admin: User = Depends(require_admin)):
    if state.router is None:
        raise HTTPException(status_code=400, detail="no model router configured")
    try:
        if getattr(state, "config_store", None) is not None:
            from app.providers import ModelCatalog

            stored = await state.config_store.load()
            state.config_revision = stored.revision
            state.router.reload(ModelCatalog(**stored.doc.models))
        else:
            state.router.reload_from_disk()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return JSONResponse(_envelope(state, reloaded=True))
