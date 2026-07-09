from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from agent.context import AgentContext, RunVars
from agent.message import from_thread
from agent.tools.base import ToolBox
from app.auth import require_user
from app.deps import AppState, get_state
from app.store.base import User
from api.protocol.chat_serializer import (
    serialize_chat_stream,
    serialize_chat_sync_with_effects,
)
from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional

router = APIRouter()


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: Optional[str] = None
    messages: List[Dict[str, Any]] = []
    stream: bool = True
    system: Optional[str] = None


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest, state: AppState = Depends(get_state),
    user: User = Depends(require_user),
):
    model = request.model or state.default_model
    msgs = from_thread(request.messages)
    if not msgs:
        raise HTTPException(
            status_code=400, detail="messages must not be empty"
        )
    current_turn = msgs[-1]
    history = msgs[:-1]
    ctx = AgentContext(
        system_prompt=request.system or "You are a helpful assistant.",
        history=history,
        current_turn=current_turn,
        attachments=[],
        hints=[],
        tools=ToolBox([]),
        run_vars=RunVars(),
    )
    agent = state.make_agent()
    events = await agent.run(ctx)
    if request.stream:

        async def gen():
            async for chunk in serialize_chat_stream(events, model=model):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")
    result = await serialize_chat_sync_with_effects(events, model=model)
    return JSONResponse(result)
