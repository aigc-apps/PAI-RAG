"""Authentication primitives + FastAPI dependencies.

Password hashing (argon2id), stateless HS256 JWTs, single-use signed invite
tokens, and the ``get_current_user`` / ``require_user`` / ``require_admin``
dependencies that gate the API. The token is accepted from either an
``Authorization: Bearer`` header (web-API clients) or the ``access_token``
httpOnly cookie (the browser); identity is always derived from the token's
``sub`` — never from a client-supplied ``user_id``.

Disable is near-real-time despite statelessness: every request re-loads the user
and rejects any status != active, so a disabled user's still-valid token stops
working within one TTL at worst (usually immediately).
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)
from fastapi import Depends, HTTPException, Request, Response

from app.config import Settings, get_settings
from app.deps import AppState, get_state
from app.store.base import User, UserAuth

ALGORITHM = "HS256"
COOKIE_NAME = "access_token"

_ph = PasswordHasher()


# --------------------------------------------------------------------------- #
# passwords
# --------------------------------------------------------------------------- #
def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password: str, password_hash: Optional[str]) -> bool:
    """Constant-time verify via argon2; False (never raises) on any mismatch or
    malformed/absent hash so callers get a uniform bad-credentials path."""
    if not password_hash:
        return False
    try:
        return _ph.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


# --------------------------------------------------------------------------- #
# JWT access tokens
# --------------------------------------------------------------------------- #
def create_access_token(user: User, *, settings: Optional[Settings] = None) -> str:
    settings = settings or get_settings()
    if not settings.jwt_secret:
        raise RuntimeError("jwt_secret is not configured (cannot issue tokens)")
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.id,
        "email": user.email,
        "role": user.role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_ttl_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_token(token: str, *, settings: Optional[Settings] = None) -> dict:
    """Decode + verify an HS256 token. Raises HTTPException(401) on
    expired/invalid, HTTPException(503) when auth is not configured."""
    settings = settings or get_settings()
    if not settings.jwt_secret:
        raise HTTPException(status_code=503, detail="authentication is not configured")
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="invalid token")


# --------------------------------------------------------------------------- #
# invite tokens (single-use, expiring; only the hash is stored)
# --------------------------------------------------------------------------- #
def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def new_invite_token() -> Tuple[str, str]:
    """(raw, hash). The raw goes into the copyable link; only the hash is
    persisted (so a DB read can't reconstruct a working invite)."""
    raw = secrets.token_urlsafe(32)
    return raw, hash_token(raw)


def invite_expiry(settings: Optional[Settings] = None) -> datetime:
    settings = settings or get_settings()
    return datetime.now(timezone.utc) + timedelta(hours=settings.invite_ttl_hours)


def invite_expired(auth: UserAuth) -> bool:
    exp = auth.invite_expires_at
    if exp is None:
        return True
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) > exp


# --------------------------------------------------------------------------- #
# cookie helpers
# --------------------------------------------------------------------------- #
def set_auth_cookie(response: Response, token: str, *, settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    response.set_cookie(
        COOKIE_NAME, token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        max_age=settings.jwt_ttl_minutes * 60,
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/")


# --------------------------------------------------------------------------- #
# FastAPI dependencies
# --------------------------------------------------------------------------- #
def _extract_token(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization") or ""
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    return request.cookies.get(COOKIE_NAME)


async def get_current_user(
    request: Request, state: AppState = Depends(get_state)
) -> Optional[User]:
    """Resolve the caller from the bearer header or cookie, or None if there is
    no/invalid token (does not raise for a missing token — use require_user to
    enforce). Re-loads the user each call so status changes take effect."""
    token = _extract_token(request)
    if not token:
        return None
    try:
        claims = decode_token(token)
    except HTTPException as exc:
        if exc.status_code == 503:
            raise
        return None
    uid = claims.get("sub")
    if not uid:
        return None
    user = await state.store.get_user(uid)
    if user is None or user.status != "active":
        return None
    return user


async def require_user(
    request: Request, state: AppState = Depends(get_state)
) -> User:
    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="authentication required")
    claims = decode_token(token)
    uid = claims.get("sub")
    user = await state.store.get_user(uid) if uid else None
    if user is None:
        raise HTTPException(status_code=401, detail="user not found")
    if user.status != "active":
        raise HTTPException(status_code=401, detail="user is not active")
    return user


async def require_admin(user: User = Depends(require_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="admin permission required")
    return user
