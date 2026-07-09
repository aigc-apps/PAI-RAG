"""Authentication + user management (`/v1/auth`).

First-run flow: ``GET /bootstrap`` reports whether any user exists; ``POST
/bootstrap`` creates the first admin (only while the user table is empty). After
that, admins invite users (``POST /invite`` returns a copyable link — there is
no email service), invitees set a password via ``POST /accept-invite``, and
everyone logs in with ``POST /login``. Identity is issued as a JWT returned in
the body (for web-API clients) and set as an httpOnly cookie (for the browser).
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from app.auth import (
    clear_auth_cookie,
    create_access_token,
    hash_password,
    hash_token,
    invite_expired,
    invite_expiry,
    new_invite_token,
    require_admin,
    require_user,
    set_auth_cookie,
    verify_password,
)
from app.config import get_settings
from app.deps import AppState, get_state
from app.store.base import User
from app.urls import external_base_url

router = APIRouter(prefix="/v1/auth", tags=["auth"])


# --------------------------------------------------------------------------- #
# payloads
# --------------------------------------------------------------------------- #
class BootstrapPayload(BaseModel):
    email: str
    password: str
    token: Optional[str] = None


class LoginPayload(BaseModel):
    email: str
    password: str


class AcceptInvitePayload(BaseModel):
    token: str
    password: str


class InvitePayload(BaseModel):
    email: str
    role: str = "user"


class StatusPayload(BaseModel):
    status: str


class ChangePasswordPayload(BaseModel):
    old_password: str
    new_password: str


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _normalize_email(email: str) -> str:
    e = (email or "").strip().lower()
    if "@" not in e or len(e) < 3:
        raise HTTPException(status_code=400, detail="a valid email is required")
    return e


def _require_password(pw: str) -> None:
    if not pw or len(pw) < 8:
        raise HTTPException(status_code=400, detail="password must be at least 8 characters")


def _user_dict(u: User) -> dict:
    return {"id": u.id, "email": u.email, "role": u.role,
            "status": u.status, "display_name": u.display_name}


def _auth_response(user: User) -> JSONResponse:
    """Issue a token, return it in the body (web-API clients) and set the
    httpOnly cookie (browser)."""
    settings = get_settings()
    if not settings.jwt_secret:
        raise HTTPException(status_code=503, detail="authentication is not configured")
    token = create_access_token(user, settings=settings)
    resp = JSONResponse({"access_token": token, "token_type": "bearer",
                         "user": _user_dict(user)})
    set_auth_cookie(resp, token, settings=settings)
    return resp


# --------------------------------------------------------------------------- #
# public endpoints
# --------------------------------------------------------------------------- #
@router.get("/bootstrap")
async def bootstrap_status(state: AppState = Depends(get_state)):
    return JSONResponse({"needed": (await state.store.count_users()) == 0})


@router.post("/bootstrap")
async def bootstrap(payload: BootstrapPayload, state: AppState = Depends(get_state)):
    if await state.store.count_users() != 0:
        raise HTTPException(status_code=409, detail="an admin account already exists")
    settings = get_settings()
    if settings.admin_bootstrap_token and payload.token != settings.admin_bootstrap_token:
        raise HTTPException(status_code=403, detail="invalid bootstrap token")
    email = _normalize_email(payload.email)
    _require_password(payload.password)
    user = await state.store.create_user(
        email=email, role="admin", status="active",
        password_hash=hash_password(payload.password),
    )
    logger.info("auth bootstrap: first admin created ({})", email)
    return _auth_response(user)


@router.post("/login")
async def login(payload: LoginPayload, state: AppState = Depends(get_state)):
    email = (payload.email or "").strip().lower()
    auth = await state.store.get_user_auth(email)
    # One generic error for not-found / wrong-password / disabled to avoid
    # account enumeration.
    if (auth is None or auth.status != "active"
            or not verify_password(payload.password, auth.password_hash)):
        raise HTTPException(status_code=401, detail="invalid email or password")
    user = await state.store.get_user(auth.id)
    return _auth_response(user)


@router.post("/logout")
async def logout():
    resp = JSONResponse({"ok": True})
    clear_auth_cookie(resp)
    return resp


@router.get("/me")
async def me(user: User = Depends(require_user)):
    return JSONResponse({"user": _user_dict(user)})


@router.post("/accept-invite")
async def accept_invite(payload: AcceptInvitePayload, state: AppState = Depends(get_state)):
    _require_password(payload.password)
    auth = await state.store.get_user_auth_by_invite(hash_token(payload.token))
    if auth is None or auth.status != "invited" or invite_expired(auth):
        raise HTTPException(status_code=400, detail="invalid or expired invite")
    user = await state.store.set_user_password(auth.id, hash_password(payload.password))
    if user is None:
        raise HTTPException(status_code=400, detail="invalid or expired invite")
    logger.info("auth accept-invite: {} activated", user.email)
    return _auth_response(user)


@router.post("/change-password")
async def change_password(
    payload: ChangePasswordPayload,
    user: User = Depends(require_user),
    state: AppState = Depends(get_state),
):
    _require_password(payload.new_password)
    auth = await state.store.get_user_auth(user.email) if user.email else None
    if auth is None or not verify_password(payload.old_password, auth.password_hash):
        raise HTTPException(status_code=400, detail="current password is incorrect")
    await state.store.set_user_password(user.id, hash_password(payload.new_password))
    return JSONResponse({"ok": True})


# --------------------------------------------------------------------------- #
# admin: user management
# --------------------------------------------------------------------------- #
@router.post("/invite")
async def invite(
    payload: InvitePayload,
    request: Request,
    admin: User = Depends(require_admin),
    state: AppState = Depends(get_state),
):
    email = _normalize_email(payload.email)
    if await state.store.get_user_by_email(email) is not None:
        raise HTTPException(status_code=409, detail="a user with that email already exists")
    role = payload.role if payload.role in ("admin", "user") else "user"
    raw, token_hash = new_invite_token()
    expires_at = invite_expiry()
    user = await state.store.create_user(
        email=email, role=role, status="invited",
        invite_token_hash=token_hash, invite_expires_at=expires_at,
    )
    invite_path = f"/?invite={raw}"
    base = external_base_url(request)
    logger.info("auth invite: {} invited as {} by {}", email, role, admin.email)
    return JSONResponse({
        "user": _user_dict(user),
        "invite_token": raw,
        "invite_path": invite_path,
        "invite_url": f"{base}{invite_path}",
        "expires_at": expires_at.isoformat(),
    })


@router.get("/users")
async def list_users(
    admin: User = Depends(require_admin),
    state: AppState = Depends(get_state),
):
    users = await state.store.list_users()
    return JSONResponse({"data": [_user_dict(u) for u in users]})


@router.post("/users/{user_id}/status")
async def set_user_status(
    user_id: str,
    payload: StatusPayload,
    admin: User = Depends(require_admin),
    state: AppState = Depends(get_state),
):
    if payload.status not in ("active", "disabled"):
        raise HTTPException(status_code=400, detail="status must be 'active' or 'disabled'")
    if user_id == admin.id and payload.status != "active":
        raise HTTPException(status_code=400, detail="you cannot disable your own account")
    user = await state.store.set_user_status(user_id, payload.status)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return JSONResponse({"user": _user_dict(user)})
