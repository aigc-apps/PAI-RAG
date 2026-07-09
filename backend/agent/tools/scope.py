from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolScope:
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    # Mostly str->str; may also carry a nested "aliyun_sandbox_env" dict that the
    # sandbox provider merges into the create-time env contract.
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_id: str = "main"
    skill_mounts: List[Dict[str, Any]] = field(default_factory=list)
    skill_fingerprint: str = "none"

    @property
    def is_admin(self) -> bool:
        raw = self.metadata.get("is_admin") or self.metadata.get("admin")
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.lower() in {"1", "true", "yes", "admin"}:
            return True
        return str(self.metadata.get("role") or "").lower() == "admin"


_current_tool_scope: ContextVar[ToolScope] = ContextVar(
    "current_tool_scope",
    default=ToolScope(),
)


def get_current_tool_scope() -> ToolScope:
    return _current_tool_scope.get()


def set_current_tool_scope(scope: ToolScope):
    return _current_tool_scope.set(scope)


def reset_current_tool_scope(token) -> None:
    _current_tool_scope.reset(token)
