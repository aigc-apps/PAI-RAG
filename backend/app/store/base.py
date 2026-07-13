from __future__ import annotations
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, List, Optional, Protocol, TypeVar

from sqlalchemy.exc import IntegrityError

# Base58 (Bitcoin/Flickr): omits 0 O I l to avoid visual/LLM ambiguity.
_ID_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_ID_SIZE = 11  # 58**11 ≈ 2**64.4 — collision-safe, DB unique index is the backstop.


def new_id(prefix: str) -> str:
    """Short, collision-safe id: ``prefix_`` + 11 Base58 chars (e.g. ``doc_7Kf9Qw2mAbc``).

    Short enough to hand to a model as a tool argument without inviting a
    mis-copied character; ``secrets.choice`` is uniform (no modulo bias)."""
    return f"{prefix}_{''.join(secrets.choice(_ID_ALPHABET) for _ in range(_ID_SIZE))}"


# Back-compat alias — existing call sites import/use ``_uuid``.
_uuid = new_id

T = TypeVar("T")


async def with_id_retry(work: Callable[[], Awaitable[T]], *, attempts: int = 3) -> T:
    """Run an async unit-of-work that builds rows with fresh ``new_id()``s and commits.

    Retry on a PK/unique collision — the DB index makes correctness guaranteed,
    not merely probable. Each attempt uses a fresh ``AsyncSession`` (a failed one
    is poisoned), so ``work`` must open its own session and regenerate its ids."""
    for attempt in range(1, attempts + 1):
        try:
            return await work()
        except IntegrityError:
            if attempt == attempts:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


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
    email: Optional[str] = None
    role: str = "user"
    status: str = "active"
    created_at: datetime = field(default_factory=_now)
    meta: dict = field(default_factory=dict)


@dataclass
class UserAuth:
    """Login-time credential view — never serialized to a client. Carries the
    password_hash so the auth service can verify without the public ``User``
    ever exposing it."""
    id: str
    email: Optional[str]
    role: str
    status: str
    password_hash: Optional[str]
    invite_token_hash: Optional[str] = None
    invite_expires_at: Optional[datetime] = None


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
    async def get_tool_result(self, conversation_id: str,
                              call_id: str) -> Optional[str]: ...
    async def save_response(self, response: StoredResponse) -> StoredResponse: ...
    async def get_response(self, response_id: str) -> Optional[StoredResponse]: ...
    async def delete_response(self, response_id: str) -> None: ...
    async def truncate_last_turn(self, conversation_id: str,
                                 response_id: str) -> Optional[str]: ...
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
    async def update_user_meta(self, user_id: str, patch: dict) -> User: ...
    # --- auth ---
    async def count_users(self) -> int: ...
    async def get_user_by_email(self, email: str) -> Optional[User]: ...
    async def get_user_auth(self, email: str) -> Optional["UserAuth"]: ...
    async def create_user(self, *, email: str, role: str, status: str,
                          password_hash: Optional[str] = None,
                          invite_token_hash: Optional[str] = None,
                          invite_expires_at: Optional[datetime] = None,
                          display_name: Optional[str] = None) -> User: ...
    async def get_user_auth_by_invite(self, invite_token_hash: str) -> Optional["UserAuth"]: ...
    async def set_user_password(self, user_id: str, password_hash: str) -> Optional[User]: ...
    async def set_user_status(self, user_id: str, status: str) -> Optional[User]: ...
    async def set_user_role(self, user_id: str, role: str) -> Optional[User]: ...
    async def list_users(self) -> List[User]: ...
    async def add_memory(self, item: MemoryItem) -> MemoryItem: ...
    async def list_memories(self, user_id: str, limit: int = 50) -> List[MemoryItem]: ...
    async def update_memory(self, memory_id: str, text: str) -> None: ...
    async def delete_memory(self, memory_id: str) -> None: ...
    async def delete_user_memories(self, user_id: str) -> None: ...
    async def update_conversation_summary(self, conversation_id: str, summary: str,
                                          summarized_seq: int) -> None: ...
