from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
from utils.time_utils import get_current_time_str
from agent.message import Message


@dataclass
class RunVars:
    """Runtime values rendered into the current turn (e.g. the time header)."""
    current_datetime: str = field(default_factory=get_current_time_str)


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
