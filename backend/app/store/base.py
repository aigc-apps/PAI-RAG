from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Protocol


def _uuid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def new_conversation_id() -> str:
    return _uuid("conv")


@dataclass
class Item:
    type: str
    content: dict
    role: Optional[str] = None
    response_id: Optional[str] = None
    id: str = field(default_factory=lambda: _uuid("item"))
    seq: int = 0


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: _uuid("conv"))
    user_id: Optional[str] = None


@dataclass
class StoredResponse:
    id: str
    model: str
    status: str
    conversation_id: Optional[str] = None
    previous_response_id: Optional[str] = None
    usage: Optional[dict] = None
    error: Optional[dict] = None


class ResponseStore(Protocol):
    async def create_conversation(self, user_id: Optional[str] = None) -> Conversation: ...
    async def append_items(self, conversation_id: str, items: List[Item]) -> List[Item]: ...
    async def get_conversation_items(self, conversation_id: str) -> List[Item]: ...
    async def save_response(self, response: StoredResponse) -> StoredResponse: ...
    async def get_response(self, response_id: str) -> Optional[StoredResponse]: ...
    async def delete_response(self, response_id: str) -> None: ...
    async def resolve_history(self, previous_response_id: Optional[str],
                              conversation: Optional[str]) -> List[Item]: ...
