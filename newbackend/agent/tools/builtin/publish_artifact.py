"""publish_artifact: surface a file the agent created under /mnt/user so the
frontend can preview (markdown/image/html/text) or download it.

The tool jails the path under /mnt/user, stats its size (via the backend NAS
mount when configured, else by reading it back from the live sandbox), signs a
stateless token, and emits a structured :class:`FileArtifact`. The string it
returns to the model carries no bytes — just a short confirmation.
"""

from __future__ import annotations

import shlex
from pathlib import Path, PurePosixPath
from typing import Optional

from agent.tools.artifacts import (
    FileArtifact,
    classify,
    emit_artifact,
    sign_artifact_token,
)
from agent.tools.base import Tool
from agent.tools.scope import get_current_tool_scope

USER_MOUNT = "/mnt/user"


def _human_size(n: int) -> str:
    if n < 0:
        return "unknown size"
    step = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if step < 1024 or unit == "GB":
            return f"{step:.0f} {unit}" if unit == "B" else f"{step:.1f} {unit}"
        step /= 1024
    return f"{n} B"


def _relativize(path: str) -> Optional[str]:
    """Return the path relative to /mnt/user, or ``None`` if it escapes the
    mount. An absolute path outside /mnt/user is rejected; a relative path is
    interpreted as relative to /mnt/user."""
    p = (path or "").strip()
    if not p:
        return None
    if p == USER_MOUNT or p.rstrip("/") == USER_MOUNT:
        return None  # the mount dir itself, not a file
    if p.startswith("/"):
        if not (p == USER_MOUNT or p.startswith(USER_MOUNT + "/")):
            return None
        rel = p[len(USER_MOUNT):]
    else:
        rel = p
    rel = rel.lstrip("/")
    if not rel:
        return None
    parts = PurePosixPath(rel).parts
    if any(part in ("..", "") for part in parts):
        return None
    return str(PurePosixPath(*parts))


def make_publish_artifact_tool(provider, settings) -> Tool:
    secret = str(getattr(settings, "files_url_secret", "") or "")
    nas_root = str(getattr(settings, "files_nas_local_root", "") or "")
    max_bytes = int(getattr(settings, "files_max_bytes", 25 * 1024 * 1024))
    path_template = str(
        getattr(provider, "nas_user_remote_path_template", "/users/{user_id}")
        or "/users/{user_id}"
    )

    async def _stat_size(user_id: str, rel: str) -> Optional[int]:
        """Best-effort byte size. Prefers the backend NAS mount; falls back to a
        `stat` in the live sandbox. Returns None when the file cannot be found."""
        if nas_root:
            user_dir = path_template.format(user_id=user_id).lstrip("/")
            local = Path(nas_root) / user_dir / rel
            try:
                if local.is_file():
                    return local.stat().st_size
            except OSError:
                pass
            return None
        try:
            sandbox_path = f"{USER_MOUNT}/{rel}"
            res = await provider.run_command(
                command=f"stat -c %s -- {shlex.quote(sandbox_path)}"
            )
        except Exception:
            return None
        if res.get("exit_code"):
            return None
        out = str(res.get("stdout") or "").strip()
        try:
            return int(out.splitlines()[0]) if out else None
        except (ValueError, IndexError):
            return None

    async def fn(path: str, name: Optional[str] = None) -> str:
        if not secret:
            return (
                "publish_artifact failed: file sharing is not configured "
                "(missing files_url_secret)."
            )
        scope = get_current_tool_scope()
        user_id = scope.user_id or ""
        if not user_id:
            return "publish_artifact failed: no user in scope; cannot publish."
        rel = _relativize(path)
        if rel is None:
            return (
                f"publish_artifact failed: {path!r} must be a file under {USER_MOUNT} "
                f"(save outputs there, e.g. {USER_MOUNT}/report.md). Paths outside "
                f"{USER_MOUNT} and '..' are not allowed."
            )
        size = await _stat_size(user_id, rel)
        if size is None:
            return (
                f"publish_artifact failed: {USER_MOUNT}/{rel} was not found. Write the "
                f"file under {USER_MOUNT} before publishing it."
            )
        display = name or PurePosixPath(rel).name
        mime, kind = classify(display)
        # Oversized files can still be downloaded but should not preview inline.
        if size > max_bytes and kind != "file":
            kind = "file"
        token = sign_artifact_token(
            user_id=user_id,
            conversation_id=scope.conversation_id or "",
            rel=rel,
            mime=mime,
            kind=kind,
            size=size,
            secret=secret,
        )
        emit_artifact(
            FileArtifact(id=token, name=display, mime=mime, size=size, kind=kind)
        )
        verb = "shown to the user for preview" if kind != "file" else "available to download"
        return f'Published "{display}" ({kind}, {_human_size(size)}). It is now {verb}.'

    return Tool(
        name="publish_artifact",
        description=(
            "Surface a file you created under /mnt/user so the user can preview or "
            "download it. Save the file under /mnt/user first (that path is durable "
            "and is exposed as $AGENT_USER_PATH in the sandbox), then call this with "
            "the path, e.g. '/mnt/user/report.md'. Markdown, images, and HTML preview "
            "in a side panel; other files get a download link."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to the file under /mnt/user, e.g. "
                        "'/mnt/user/chart.png' or 'outputs/report.md'."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": "Optional display/download name; defaults to the file's basename.",
                },
            },
            "required": ["path"],
        },
        fn=fn,
    )
