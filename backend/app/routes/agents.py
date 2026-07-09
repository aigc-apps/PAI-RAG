"""User-facing agent roster.

Unlike the agent-config document (``GET /v1/setup``, admin-only), this exposes
just the selectable agents — id / name / description / model — to ANY
authenticated user, so the chat UI can offer an agent switcher without granting
non-admins access to the control plane. Managing (create/edit) agents stays
admin-only via the config routes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth import require_user
from app.deps import AppState, get_state
from app.store.base import User

router = APIRouter()


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
