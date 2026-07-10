"""User-facing agent roster.

Unlike the agent-config document (``GET /v1/setup``, admin-only), this exposes
just the selectable agents — id / name / description / model — to ANY
authenticated user, so the chat UI can offer an agent switcher without granting
non-admins access to the control plane. Managing (create/edit) agents stays
admin-only via the config routes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from agent.core.events import RunFailed, TextDelta
from app.auth import require_admin, require_user
from app.builder import build_context, resolve_agent_model
from app.deps import AppState, get_state
from app.schemas import ResponsesRequest
from app.store.base import User

router = APIRouter()

# What the model is asked to do when generating a code manifest: drive the
# sandbox tools itself to look at /mnt/code, then emit only the finished list.
_MANIFEST_INSTRUCTION = (
    "You are documenting the source-code repositories mounted read-only at "
    "/mnt/code, for another AI agent's system prompt. Explore them yourself "
    "with the shell / code_interpreter tools:\n"
    "1. Run `ls /mnt/code` — each subdirectory is one repository.\n"
    "2. For each repository, read its README and skim its top-level layout "
    "(ls, cat the README and obvious entry files) to learn what it is for.\n\n"
    "Then output ONLY a concise Markdown manifest: one short section or bullet "
    "per repository, giving its directory name under /mnt/code and a 1-2 "
    "sentence description of what it contains and when it would be relevant to "
    "consult. Do NOT include your shell transcript or exploration steps — just "
    "the final manifest. If /mnt/code is empty or unreadable, say so in one line."
)

# Cap the exploration loop so a manifest generation can't run the full 20-step
# budget; a handful of ls/cat rounds is plenty.
_MANIFEST_MAX_STEPS = 8


def _err(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message}})


@router.get("/v1/agents")
async def list_agents(
    state: AppState = Depends(get_state),
    _user: User = Depends(require_user),
):
    doc = getattr(state, "agent_config", None)
    agents = list(getattr(doc, "agents", []) or []) if doc is not None else []
    if not agents:
        # No config document (or none defined) → surface a single default agent so
        # the switcher always has an entry matching the live soul.
        name = getattr(getattr(state, "soul", None), "name", "MiniAgent") or "MiniAgent"
        return {"agents": [{"id": "main", "name": name, "description": "", "model": ""}], "default_agent": "main"}
    return {
        "agents": [
            {
                "id": a.id,
                "name": a.name,
                "description": getattr(a, "description", "") or "",
                "model": getattr(a, "model", "") or "",
            }
            for a in agents
        ],
        "default_agent": getattr(doc, "default_agent", "main") or "main",
    }


@router.post("/v1/agents/{agent_id}/code-manifest/generate")
async def generate_code_manifest(
    agent_id: str,
    state: AppState = Depends(get_state),
    _admin: User = Depends(require_admin),
):
    """Have the LLM explore the read-only /mnt/code layer (via the sandbox tools)
    and return a Markdown manifest of the repositories. Admin-only authoring
    action; the result is NOT persisted — the caller reviews it and saves it onto
    the agent profile through the normal config save."""
    # The code layer must actually be mounted, or there's nothing to document.
    provider = getattr(getattr(state, "registry", None), "sandbox_provider", None)
    if provider is None or not getattr(provider, "nas_code_server_addr", ""):
        return _err(409, "code layer not configured (sandbox /mnt/code is unavailable)")

    doc = getattr(state, "agent_config", None)
    agents = list(getattr(doc, "agents", []) or []) if doc is not None else []
    if agents and not any(a.id == agent_id for a in agents):
        return _err(404, f"unknown agent: {agent_id}")

    if state.router is None:
        return _err(503, "no model provider configured")

    request = ResponsesRequest(
        agent_id=agent_id, input=_MANIFEST_INSTRUCTION,
        store=False, stream=False, background=False, memory=False,
    )
    candidate = resolve_agent_model(state.agent_config, request)
    if candidate:
        try:
            state.router.get_config(candidate)
        except KeyError:
            candidate = None
    request.model = candidate or state.router.default_model_id
    try:
        cfg = state.router.get_config(request.model)
    except KeyError:
        return _err(404, f"unknown model: {request.model}")
    try:
        llm = state.router.get_llm(request.model)
    except RuntimeError as e:
        return _err(503, str(e))

    # store=False + no conversation/previous_response_id => build_context mints an
    # ephemeral conversation id but writes nothing; we never call _persist, so no
    # chat history is created by generation.
    try:
        ctx, _conv = await build_context(
            request, state.store, soul=state.soul, registry=state.registry,
            agent_config=state.agent_config,
            project_context=getattr(state, "project_context", ""),
            authenticated_user_id=_admin.id,
        )
    except ValueError as e:
        return _err(400, str(e))

    agent = state.make_agent(
        llm=llm, context_window=cfg.context_window,
        max_output_tokens=cfg.max_output_tokens,
    )
    agent.max_steps = _MANIFEST_MAX_STEPS

    chunks: list[str] = []
    async for ev in await agent.run(ctx):
        if isinstance(ev, TextDelta):
            chunks.append(ev.text)
        elif isinstance(ev, RunFailed):
            return _err(502, f"manifest generation failed: {ev.message}")
    manifest = "".join(chunks).strip()
    return {"manifest": manifest}
