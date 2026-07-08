"""Byte-serving for sandbox file artifacts.

``GET /v1/files/{artifact_id}?user_id=…`` resolves an HMAC-signed token to bytes.
Primary path: read from the NAS export the backend host has mounted at
``settings.files_nas_local_root`` (the same export the sandbox sees at
/mnt/user), path-jailed to the user's subdirectory. Fallback (no backend mount):
read the file back from the live conversation sandbox via ``base64`` — best
effort, since it needs a still-warm session.
"""

from __future__ import annotations

import base64
import shlex
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from agent.tools.artifacts import (
    PREVIEW_KINDS,
    ArtifactTokenError,
    verify_artifact_token,
)
from agent.tools.scope import (
    ToolScope,
    reset_current_tool_scope,
    set_current_tool_scope,
)
from app.auth import require_user
from app.config import get_settings
from app.deps import AppState, get_state
from app.store.base import User

router = APIRouter()


def _content_disposition(kind: str, name: str) -> str:
    disposition = "inline" if kind in PREVIEW_KINDS else "attachment"
    # Quote the filename defensively; strip characters that would break the header.
    safe = name.replace('"', "").replace("\r", "").replace("\n", "")
    return f'{disposition}; filename="{safe}"'


@router.get("/v1/files/{artifact_id}")
async def get_file(
    artifact_id: str,
    state: AppState = Depends(get_state),
    user: User = Depends(require_user),
):
    settings = get_settings()
    secret = str(getattr(settings, "files_url_secret", "") or "")
    if not secret:
        raise HTTPException(status_code=503, detail="file sharing is not configured")
    try:
        claim = verify_artifact_token(artifact_id, secret=secret)
    except ArtifactTokenError:
        raise HTTPException(status_code=403, detail="invalid file token")
    # Identity comes from the authenticated session (cookie flows on <img> loads),
    # not a client-supplied query param — the token must belong to the caller.
    if not claim.user_id or claim.user_id != user.id:
        raise HTTPException(status_code=403, detail="file token does not match user")

    max_bytes = int(getattr(settings, "files_max_bytes", 25 * 1024 * 1024))
    headers = {"Content-Disposition": _content_disposition(claim.kind, _basename(claim.rel))}
    nas_root = str(getattr(settings, "files_nas_local_root", "") or "")

    if nas_root:
        return _serve_from_nas(
            nas_root=nas_root,
            claim=claim,
            provider=getattr(state.registry, "sandbox_provider", None),
            max_bytes=max_bytes,
            headers=headers,
        )
    return await _serve_from_sandbox(
        provider=getattr(state.registry, "sandbox_provider", None),
        claim=claim,
        max_bytes=max_bytes,
        headers=headers,
    )


def _basename(rel: str) -> str:
    return rel.rsplit("/", 1)[-1] if rel else "download"


def _serve_from_nas(*, nas_root, claim, provider, max_bytes, headers) -> FileResponse:
    template = str(
        getattr(provider, "nas_user_remote_path_template", "/users/{user_id}")
        or "/users/{user_id}"
    )
    user_dir = template.format(user_id=claim.user_id).lstrip("/")
    base = (Path(nas_root) / user_dir).resolve()
    try:
        target = (base / claim.rel).resolve()
    except (OSError, ValueError):
        raise HTTPException(status_code=403, detail="invalid file path")
    # Path jail: the resolved target must stay inside the user's NAS subdirectory.
    if target != base and base not in target.parents:
        raise HTTPException(status_code=403, detail="file path escapes user directory")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    if target.stat().st_size > max_bytes:
        raise HTTPException(status_code=413, detail="file too large")
    return FileResponse(target, media_type=claim.mime, headers=headers)


async def _serve_from_sandbox(*, provider, claim, max_bytes, headers) -> Response:
    if provider is None:
        raise HTTPException(status_code=503, detail="no sandbox available to read file")
    if claim.size is not None and claim.size > max_bytes:
        raise HTTPException(status_code=413, detail="file too large")
    sandbox_path = f"/mnt/user/{claim.rel}"
    # Reconstruct the tool scope so the provider reuses the same warm session that
    # produced the file. Best-effort: a different agent/skill fingerprint or an
    # idle-reaped sandbox means the read misses.
    token = set_current_tool_scope(
        ToolScope(user_id=claim.user_id, conversation_id=claim.conversation_id)
    )
    try:
        res = await provider.run_command(
            command=f"base64 -w0 -- {shlex.quote(sandbox_path)}"
        )
    except Exception:
        raise HTTPException(status_code=410, detail="sandbox unavailable")
    finally:
        reset_current_tool_scope(token)
    if res.get("exit_code"):
        raise HTTPException(status_code=404, detail="file not found in sandbox")
    encoded = str(res.get("stdout") or "").strip()
    try:
        raw = base64.b64decode(encoded, validate=False)
    except Exception:
        raise HTTPException(status_code=502, detail="could not decode file bytes")
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="file too large")
    return Response(content=raw, media_type=claim.mime, headers=headers)
