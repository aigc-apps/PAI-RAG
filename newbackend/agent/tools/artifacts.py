"""Sandbox file artifacts: the shared contract, an ext→(mime,kind) classifier,
a contextvar collector that bridges plain-string tool returns to structured
output, and a stateless HMAC-signed token codec.

No FastAPI/store dependency here so both the tool layer (producer) and the route
layer (byte server) can import it. A `FileArtifact` is what the frontend sees
(``{id, name, mime, size, kind}``); the real sandbox path lives ONLY inside the
signed token, never serialized outward.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import mimetypes
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel

Kind = Literal["image", "markdown", "html", "text", "file"]

# Preview-capable kinds render inline in the right-side panel; everything else
# is download-only.
PREVIEW_KINDS = frozenset({"image", "markdown", "html", "text"})


class FileArtifact(BaseModel):
    """Outward-facing artifact reference. ``id`` is the signed token the browser
    GETs from ``/v1/files/{id}``; it opaquely carries the path + scope."""

    id: str
    name: str
    mime: str
    size: int  # bytes; -1 when unknown (readback fallback couldn't stat)
    kind: Kind


# ---------------------------------------------------------------------------
# Classification: extension -> (mime, kind)
# ---------------------------------------------------------------------------

_IMAGE = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico"}
_MARKDOWN = {".md", ".markdown"}
_HTML = {".html", ".htm"}
_TEXT = {
    ".txt", ".csv", ".tsv", ".json", ".log", ".yaml", ".yml", ".xml",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".css", ".ini", ".toml",
}


def classify(name: str) -> Tuple[str, Kind]:
    """Map a file name to ``(mime, kind)``. SVG is deliberately treated as a
    plain ``file`` (attachment) rather than ``image`` to avoid inline-SVG XSS in
    the preview panel."""
    ext = PurePosixPath(name).suffix.lower()
    guessed, _ = mimetypes.guess_type(name)
    if ext in _IMAGE:
        return guessed or f"image/{ext.lstrip('.') or 'png'}", "image"
    if ext in _MARKDOWN:
        return "text/markdown", "markdown"
    if ext in _HTML:
        return "text/html", "html"
    if ext in _TEXT:
        return guessed or "text/plain", "text"
    return guessed or "application/octet-stream", "file"


# ---------------------------------------------------------------------------
# Contextvar collector: tools emit artifacts, dispatch drains them
# ---------------------------------------------------------------------------

_pending: ContextVar[Optional[List[FileArtifact]]] = ContextVar(
    "pending_artifacts", default=None
)


def begin_artifact_capture() -> Token:
    """Start a fresh capture buffer for one tool call. Returned token is passed
    back to :func:`end_artifact_capture`."""
    return _pending.set([])


def emit_artifact(artifact: FileArtifact) -> None:
    """Record an artifact produced by the currently running tool. A no-op if no
    capture is active (tool called outside dispatch)."""
    buf = _pending.get()
    if buf is not None:
        buf.append(artifact)


def drain_artifacts() -> List[FileArtifact]:
    """Return and clear the artifacts captured for the current tool call."""
    buf = _pending.get()
    if not buf:
        return []
    out = list(buf)
    buf.clear()
    return out


def end_artifact_capture(token: Token) -> None:
    _pending.reset(token)


# ---------------------------------------------------------------------------
# Contextvar collector: a single structured "notice" a tool can surface beside
# its string return (e.g. "aliyun authorization required"). Unlike artifacts it
# is a single slot (last write wins) and is stream-only — never persisted — so a
# reloaded conversation doesn't replay a stale prompt.
# ---------------------------------------------------------------------------

_pending_notice: ContextVar[Optional[dict]] = ContextVar("pending_tool_notice", default=None)


def begin_notice_capture() -> Token:
    return _pending_notice.set(None)


def emit_tool_notice(notice: dict) -> None:
    """Record a structured notice for the currently running tool call. Only the
    most recent one survives; a no-op outside an active capture."""
    _pending_notice.set(notice)


def drain_tool_notice() -> Optional[dict]:
    """Return and clear the notice captured for the current tool call."""
    notice = _pending_notice.get()
    if notice is not None:
        _pending_notice.set(None)
    return notice


def end_notice_capture(token: Token) -> None:
    _pending_notice.reset(token)


# ---------------------------------------------------------------------------
# Stateless signed token: artifact id <-> (scope + path) claim
# ---------------------------------------------------------------------------


@dataclass
class ArtifactClaim:
    user_id: str
    conversation_id: str
    rel: str  # path relative to /mnt/user, no leading slash, no '..'
    mime: str
    kind: str
    size: int


class ArtifactTokenError(ValueError):
    """Raised when a token is malformed, tampered, or signed with another key."""


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def sign_artifact_token(
    *,
    user_id: str,
    conversation_id: str,
    rel: str,
    mime: str,
    kind: str,
    size: int,
    secret: str,
) -> str:
    """Encode a claim into ``base64url(json).base64url(hmac)``. No expiry: the
    file's lifetime on NAS is the real bound, so reloaded old conversations still
    resolve as long as the secret is unchanged."""
    if not secret:
        raise ArtifactTokenError("artifact signing secret is not configured")
    payload = {
        "v": 1,
        "u": user_id,
        "c": conversation_id,
        "r": rel,
        "m": mime,
        "k": kind,
        "s": size,
    }
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return f"{_b64e(body)}.{_b64e(sig)}"


def verify_artifact_token(token: str, *, secret: str) -> ArtifactClaim:
    if not secret:
        raise ArtifactTokenError("artifact signing secret is not configured")
    try:
        body_b64, sig_b64 = token.split(".", 1)
        body = _b64d(body_b64)
        got_sig = _b64d(sig_b64)
    except Exception as ex:  # malformed structure
        raise ArtifactTokenError(f"malformed artifact token: {ex}") from ex
    want_sig = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    if not hmac.compare_digest(got_sig, want_sig):
        raise ArtifactTokenError("artifact token signature mismatch")
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as ex:
        raise ArtifactTokenError(f"artifact token payload not JSON: {ex}") from ex
    return ArtifactClaim(
        user_id=str(payload.get("u") or ""),
        conversation_id=str(payload.get("c") or ""),
        rel=str(payload.get("r") or ""),
        mime=str(payload.get("m") or "application/octet-stream"),
        kind=str(payload.get("k") or "file"),
        size=int(payload.get("s") if payload.get("s") is not None else -1),
    )
