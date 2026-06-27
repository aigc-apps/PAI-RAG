from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Protocol


def _uuid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def new_conversation_id() -> str:
    return _uuid("conv")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Item:
    type: str
    content: dict
    role: Optional[str] = None
    response_id: Optional[str] = None
    user_id: Optional[str] = None
    id: str = field(default_factory=lambda: _uuid("item"))
    seq: int = 0


@dataclass
class User:
    id: str
    display_name: Optional[str] = None
    created_at: datetime = field(default_factory=_now)
    meta: dict = field(default_factory=dict)


@dataclass
class Conversation:
    id: str = field(default_factory=lambda: _uuid("conv"))
    user_id: Optional[str] = None
    title: Optional[str] = None
    last_response_id: Optional[str] = None
    summary: Optional[str] = None
    summarized_seq: int = -1
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


@dataclass
class StoredResponse:
    id: str
    model: str
    status: str
    conversation_id: Optional[str] = None
    previous_response_id: Optional[str] = None
    usage: Optional[dict] = None
    error: Optional[dict] = None


@dataclass
class MemoryItem:
    user_id: str
    text: str
    kind: str = "fact"
    source_response_id: Optional[str] = None
    status: str = "active"
    id: str = field(default_factory=lambda: _uuid("mem"))
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


class ResponseStore(Protocol):
    async def create_conversation(self, user_id: Optional[str] = None) -> Conversation: ...
    async def append_items(self, conversation_id: str, items: List[Item]) -> List[Item]: ...
    async def get_conversation_items(self, conversation_id: str) -> List[Item]: ...
    async def save_response(self, response: StoredResponse) -> StoredResponse: ...
    async def get_response(self, response_id: str) -> Optional[StoredResponse]: ...
    async def delete_response(self, response_id: str) -> None: ...
    async def resolve_history(self, previous_response_id: Optional[str],
                              conversation: Optional[str]) -> List[Item]: ...
    async def ensure_conversation(self, conversation_id: str, user_id: Optional[str],
                                  title: Optional[str]) -> Conversation: ...
    async def touch_conversation(self, conversation_id: str, last_response_id: str) -> None: ...
    async def list_conversations(self, user_id: Optional[str], limit: int = 50,
                                 offset: int = 0) -> List[Conversation]: ...
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]: ...
    async def list_conversation_responses(self, conversation_id: str) -> List[StoredResponse]: ...
    async def delete_conversation(self, conversation_id: str) -> None: ...
    async def ensure_user(self, user_id: str, display_name: Optional[str] = None) -> User: ...
    async def get_user(self, user_id: str) -> Optional[User]: ...
    async def add_memory(self, item: MemoryItem) -> MemoryItem: ...
    async def list_memories(self, user_id: str, limit: int = 50) -> List[MemoryItem]: ...
    async def update_memory(self, memory_id: str, text: str) -> None: ...
    async def delete_memory(self, memory_id: str) -> None: ...
    async def delete_user_memories(self, user_id: str) -> None: ...
    async def update_conversation_summary(self, conversation_id: str, summary: str,
                                          summarized_seq: int) -> None: ...
