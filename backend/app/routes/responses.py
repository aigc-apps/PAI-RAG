from __future__ import annotations
import asyncio
import time
import uuid
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from app.schemas import ResponsesRequest
from app.auth import require_user
from app.builder import build_context, resolve_agent_model
from app.deps import AppState, get_state
from app.store.base import Item, StoredResponse, User
from app.memory import update_user_memory, make_complete
from app.summarizer import maybe_summarize_conversation
from api.protocol.responses_serializer import (
    serialize_response_sync,
    serialize_response_stream,
    make_failed_sse,
)

router = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def _error(status: int, message: str):
    """OpenAI-shaped error JSON response (``{"error": {message, type, ...}}``)."""
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "server_error",
                "param": None,
                "code": None,
            }
        },
    )


def _rid() -> str:
    return f"resp_{uuid.uuid4().hex}"


async def _owns_response(state: AppState, stored: StoredResponse, user: User) -> bool:
    """A response belongs to whoever owns its conversation. Ownership is strict
    even for admins — admin authority is over the control plane, not other users'
    private chats. A response with no conversation can't be attributed, so it is
    treated as not owned."""
    if not stored.conversation_id:
        return False
    conv = await state.store.get_conversation(stored.conversation_id)
    return conv is not None and conv.user_id == user.id


def _user_input_items(current_turn, response_id: str, user_id=None) -> list:
    """The user's turn, stored as a message item (history source of truth).

    Uses the exact text the agent saw (``current_turn.content``) rather than
    re-deriving it from the raw request, so persisted history can never drift
    from what the model was actually given.
    """
    text = current_turn.content if isinstance(current_turn.content, str) else ""
    return [Item(type="message", role="user", content={"text": text},
                 response_id=response_id, user_id=user_id)]


def _title_from_turn(current_turn) -> str:
    text = current_turn.content if isinstance(current_turn.content, str) else ""
    return text.strip()[:80]


async def _persist(
    state: AppState,
    request: ResponsesRequest,
    current_turn,
    response_id: str,
    conversation_id: str,
    store_items: list,
    status: str,
    usage: dict | None,
    error: dict | None = None,
    *,
    uid: str | None = None,
):
    if uid:
        await state.store.ensure_user(uid)
    await state.store.ensure_conversation(
        conversation_id, user_id=uid, title=_title_from_turn(current_turn),
    )
    items = _user_input_items(current_turn, response_id, user_id=uid)
    for d in store_items:
        items.append(Item(type=d["type"], role=d.get("role"), content=d["content"],
                          response_id=response_id, user_id=uid))
    await state.store.append_items(conversation_id, items)
    await state.store.save_response(
        StoredResponse(
            id=response_id,
            conversation_id=conversation_id,
            model=request.model or state.default_model,
            status=status,
            usage=usage,
            error=error,
            previous_response_id=request.previous_response_id,
        )
    )
    # Bump updated_at + record the latest response as the continuation anchor.
    await state.store.touch_conversation(conversation_id, last_response_id=response_id)


def _schedule_memory_update(state, request, current_turn, store_items, response_id, uid=None):
    if not (getattr(state, "memory_enabled", False) and request.memory and uid):
        return
    user_text = current_turn.content if isinstance(current_turn.content, str) else ""
    assistant_text = ""
    for d in store_items:
        if d.get("type") == "message" and d.get("role") == "assistant":
            assistant_text = (d.get("content") or {}).get("text", "")
    # Resolve the LLM used for memory extraction: memory_model (if routed) else the
    # request's model client (router) else state.llm.
    llm = None
    if state.router is not None:
        model_id = getattr(state, "memory_model", "") or request.model
        try:
            llm = state.router.get_llm(model_id)
        except Exception:
            llm = None
    if llm is None:
        llm = state.llm
    if llm is None:
        return
    asyncio.create_task(update_user_memory(
        state.store, uid, user_text, assistant_text, make_complete(llm),
        source_response_id=response_id,
    ))


def _schedule_summary(state, request, conversation_id):
    if not (getattr(state, "summary_enabled", False) and conversation_id):
        return
    llm = None
    if state.router is not None:
        try:
            llm = state.router.get_llm(getattr(state, "memory_model", "") or request.model)
        except Exception:
            llm = None
    llm = llm or state.llm
    if llm is None:
        return
    asyncio.create_task(maybe_summarize_conversation(
        state.store, conversation_id, make_complete(llm),
        keep_recent=getattr(state, "summary_keep_recent", 20),
        batch=getattr(state, "summary_batch", 20),
    ))


@router.post("/v1/responses")
async def create_response(
    request: ResponsesRequest,
    req: Request,
    state: AppState = Depends(get_state),
    user: User = Depends(require_user),
):
    if not request.model:
        # Prefer the selected agent's pinned model, but only when the router can
        # actually serve it — a profile referencing a since-removed model must not
        # 404 the request; fall back to the router/app default instead.
        candidate = resolve_agent_model(state.agent_config, request)
        if candidate and state.router is not None:
            try:
                state.router.get_config(candidate)
            except KeyError:
                candidate = None
        request.model = candidate or (
            state.router.default_model_id if state.router is not None else state.default_model
        )

    cfg = None
    if state.router is not None:
        try:
            cfg = state.router.get_config(request.model)
        except KeyError:
            return _error(404, f"unknown model: {request.model}")

    tools_ok = cfg.supports_tools if cfg is not None else True
    try:
        # build_context is side-effect-free: conversation_id is ALWAYS non-None
        # (freshly minted for new turns, or the resolved existing one when linking).
        ctx, conversation_id = await build_context(
            request, state.store, soul=state.soul,
            registry=(state.registry if tools_ok else None),
            agent_config=state.agent_config,
            project_context=getattr(state, "project_context", ""),
            authenticated_user_id=user.id,
        )
    except ValueError as e:
        return _error(400, str(e))

    response_id = _rid()
    if state.router is not None and cfg is not None:
        try:
            llm = state.router.get_llm(request.model)
        except RuntimeError as e:
            return _error(503, str(e))
        agent = state.make_agent(
            llm=llm,
            context_window=cfg.context_window,
            max_output_tokens=cfg.max_output_tokens,
        )
    else:
        agent = state.make_agent()
    events = await agent.run(ctx)

    if request.background:
        async def _persist_run(sink: dict, status: str):
            if not request.store:
                return
            r = sink["response"]
            await _persist(
                state, request, ctx.current_turn, response_id, conversation_id,
                sink["items"], status, r.get("usage"), r.get("error"), uid=user.id,
            )
            _schedule_memory_update(state, request, ctx.current_turn, sink["items"], response_id, user.id)
            _schedule_summary(state, request, conversation_id)

        run = state.runs.start(
            events=events, model=request.model, response_id=response_id,
            conversation_id=conversation_id, persist=_persist_run, user_id=user.id,
        )
        if request.stream:
            return StreamingResponse(
                state.runs.subscribe(run, starting_after=0),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        return JSONResponse(
            {
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "model": request.model,
                "conversation": {"id": conversation_id},
                "created_at": time.time(),
            }
        )

    if request.stream:

        async def gen():
            sink = {}
            try:
                async for chunk in serialize_response_stream(
                    events,
                    model=request.model,
                    response_id=response_id,
                    conversation_id=conversation_id,
                    sink=sink,
                ):
                    yield chunk
            except Exception as e:
                yield make_failed_sse(
                    response_id, request.model, conversation_id, str(e),
                )
                if not sink.get("response"):
                    sink["response"] = {
                        "id": response_id,
                        "status": "failed",
                        "error": {"code": "server_error", "message": str(e)},
                        "usage": None,
                    }
                    sink["items"] = []
            if request.store and sink.get("response"):
                r = sink["response"]
                await _persist(
                    state,
                    request,
                    ctx.current_turn,
                    response_id,
                    conversation_id,
                    sink["items"],
                    r["status"],
                    r.get("usage"),
                    r.get("error"),
                    uid=user.id,
                )
                _schedule_memory_update(state, request, ctx.current_turn, sink["items"], response_id, user.id)
                _schedule_summary(state, request, conversation_id)

        return StreamingResponse(gen(), media_type="text/event-stream", headers=_SSE_HEADERS)

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
            ctx.current_turn,
            response_id,
            conversation_id,
            store_items,
            resp_dict["status"],
            resp_dict.get("usage"),
            resp_dict.get("error"),
            uid=user.id,
        )
        _schedule_memory_update(state, request, ctx.current_turn, store_items, response_id, user.id)
        _schedule_summary(state, request, conversation_id)
    return JSONResponse(resp_dict)


@router.get("/v1/responses/{response_id}")
async def get_response(
    response_id: str,
    stream: bool = False,
    starting_after: int = 0,
    state: AppState = Depends(get_state),
    user: User = Depends(require_user),
):
    if stream:
        run = state.runs.get(response_id)
        if run is None:
            raise HTTPException(status_code=409, detail="run not resumable")
        if run.user_id != user.id:
            # Don't leak another user's live stream; 409 mirrors the not-found path.
            raise HTTPException(status_code=409, detail="run not resumable")
        return StreamingResponse(
            state.runs.subscribe(run, starting_after=starting_after),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    stored = await state.store.get_response(response_id)
    if stored is None or not await _owns_response(state, stored, user):
        raise HTTPException(status_code=404, detail="response not found")
    return JSONResponse(
        {
            "id": stored.id,
            "object": "response",
            "status": stored.status,
            "model": stored.model,
            "conversation": (
                {"id": stored.conversation_id} if stored.conversation_id else None
            ),
            "usage": stored.usage,
            "error": stored.error,
            "previous_response_id": stored.previous_response_id,
        }
    )


@router.delete("/v1/responses/{response_id}")
async def delete_response(
    response_id: str, state: AppState = Depends(get_state),
    user: User = Depends(require_user),
):
    stored = await state.store.get_response(response_id)
    if stored is None or not await _owns_response(state, stored, user):
        raise HTTPException(status_code=404, detail="response not found")
    await state.store.delete_response(response_id)
    return JSONResponse(
        {"id": response_id, "object": "response.deleted", "deleted": True}
    )


@router.post("/v1/responses/{response_id}/cancel")
async def cancel_response(response_id: str, state: AppState = Depends(get_state),
                          user: User = Depends(require_user)):
    run = state.runs.get(response_id)
    if run is not None:
        if run.user_id != user.id:
            raise HTTPException(status_code=404, detail="response not found")
        if await state.runs.cancel(response_id):
            return JSONResponse(
                {"id": response_id, "object": "response.cancel", "status": "cancelling"}
            )
    # no live run: report the stored status if we have it, else 404
    stored = await state.store.get_response(response_id)
    if stored is None or not await _owns_response(state, stored, user):
        raise HTTPException(status_code=404, detail="response not found")
    return JSONResponse(
        {"id": response_id, "object": "response.cancel", "status": stored.status}
    )
