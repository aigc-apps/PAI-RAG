from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from utils.time_utils import get_current_date_str, get_local_timezone_name
from agent.message import Message


@dataclass
class RunVars:
    """Slow-changing environment values appended to the stable system prompt."""
    current_date: str = field(default_factory=get_current_date_str)
    timezone: str = field(default_factory=get_local_timezone_name)


@dataclass
class Attachment:
    """Resolved attachment content to inject inline (file text or a status note)."""
    name: str
    body: str


@dataclass
class AgentContext:
    """Everything needed to build model input and run, assembled ONCE by the caller.
    This is the single inspectable object: log it and you know what the model sees."""
    system_prompt: str
    history: List[Message]
    current_turn: Message
    attachments: List[Attachment]
    hints: List[str]
    tools: object  # ToolBox; typed loosely to avoid an import cycle
    run_vars: RunVars
    context_block: str = ""
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    # Mostly str->str, but may carry a nested value (e.g. "aliyun_sandbox_env":
    # {ALIBABACLOUD_*}) that the sandbox provider reads at create time.
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_id: str = "main"
    skill_mounts: List[Dict[str, Any]] = field(default_factory=list)
    skill_fingerprint: str = "none"
