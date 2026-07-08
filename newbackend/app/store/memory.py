from __future__ import annotations
from typing import Dict, List, Optional
from app.store.base import Conversation, Item, MemoryItem, StoredResponse, User, UserAuth, _now, _uuid


class InMemoryStore:
    def __init__(self):
        self._convs: Dict[str, Conversation] = {}
        self._items: Dict[str, List[Item]] = {}
        self._responses: Dict[str, StoredResponse] = {}
        self._users: Dict[str, User] = {}
        # user_id -> credential view (password_hash / invite token), kept out of
        # the public User so it never leaks through get_user.
        self._auth: Dict[str, UserAuth] = {}
        self._memories: Dict[str, MemoryItem] = {}

    async def create_conversation(self, user_id: Optional[str] = None) -> Conversation:
        conv = Conversation(user_id=user_id)
        self._convs[conv.id] = conv
        self._items[conv.id] = []
        return conv

    async def append_items(self, conversation_id: str, items: List[Item]) -> List[Item]:
        log = self._items.setdefault(conversation_id, [])
        # Mirror SqlStore: next seq is max(seq)+1, so the contract is identical
        # (monotonic, never reused) even if gaps ever appear.
        n = max((it.seq for it in log), default=-1) + 1
        for it in items:
            it.seq = n
            log.append(it)
            n += 1
        return items

    async def get_conversation_items(self, conversation_id: str) -> List[Item]:
        return list(self._items.get(conversation_id, []))

    async def save_response(self, response: StoredResponse) -> StoredResponse:
        self._responses[response.id] = response
        return response

    async def get_response(self, response_id: str) -> Optional[StoredResponse]:
        return self._responses.get(response_id)

    async def delete_response(self, response_id: str) -> None:
        self._responses.pop(response_id, None)

    async def resolve_history(self, previous_response_id, conversation) -> List[Item]:
        conv_id = conversation
        if previous_response_id:
            resp = self._responses.get(previous_response_id)
            if resp is None:
                return []
            if conversation and resp.conversation_id != conversation:
                raise ValueError("previous_response_id does not belong to conversation")
            conv_id = resp.conversation_id
        if not conv_id:
            return []
        return list(self._items.get(conv_id, []))

    async def ensure_conversation(self, conversation_id, user_id, title) -> Conversation:
        existing = self._convs.get(conversation_id)
        if existing is not None:
            return existing
        conv = Conversation(id=conversation_id, user_id=user_id, title=title)
        self._convs[conversation_id] = conv
        self._items.setdefault(conversation_id, [])
        return conv

    async def touch_conversation(self, conversation_id, last_response_id) -> None:
        conv = self._convs.get(conversation_id)
        if conv is None:
            return
        conv.last_response_id = last_response_id
        conv.updated_at = _now()

    async def list_conversations(self, user_id, limit=50, offset=0) -> List[Conversation]:
        convs = [c for c in self._convs.values()
                 if user_id is None or c.user_id == user_id]
        convs.sort(key=lambda c: c.updated_at, reverse=True)
        return convs[offset:offset + limit]

    async def get_conversation(self, conversation_id) -> Optional[Conversation]:
        return self._convs.get(conversation_id)

    async def list_conversation_responses(self, conversation_id) -> List[StoredResponse]:
        return [r for r in self._responses.values()
                if r.conversation_id == conversation_id]

    async def ensure_user(self, user_id, display_name=None) -> User:
        u = self._users.get(user_id)
        if u is not None:
            return u
        u = User(id=user_id, display_name=display_name)
        self._users[user_id] = u
        return u

    async def get_user(self, user_id) -> Optional[User]:
        return self._users.get(user_id)

    async def update_user_meta(self, user_id, patch) -> User:
        u = self._users.get(user_id)
        if u is None:
            u = User(id=user_id)
            self._users[user_id] = u
        meta = dict(u.meta or {})
        meta.update(patch)
        u.meta = meta
        return u

    # --- auth ---
    async def count_users(self) -> int:
        return len(self._users)

    async def get_user_by_email(self, email: str) -> Optional[User]:
        return next((u for u in self._users.values() if u.email == email), None)

    async def get_user_auth(self, email: str) -> Optional[UserAuth]:
        u = await self.get_user_by_email(email)
        return self._auth.get(u.id) if u is not None else None

    async def get_user_auth_by_invite(self, invite_token_hash: str) -> Optional[UserAuth]:
        return next((a for a in self._auth.values()
                     if a.invite_token_hash == invite_token_hash), None)

    async def create_user(self, *, email, role, status, password_hash=None,
                          invite_token_hash=None, invite_expires_at=None,
                          display_name=None) -> User:
        uid = _uuid("user")
        u = User(id=uid, display_name=display_name, email=email, role=role, status=status)
        self._users[uid] = u
        self._auth[uid] = UserAuth(id=uid, email=email, role=role, status=status,
                                   password_hash=password_hash,
                                   invite_token_hash=invite_token_hash,
                                   invite_expires_at=invite_expires_at)
        return u

    async def set_user_password(self, user_id: str, password_hash: str) -> Optional[User]:
        u = self._users.get(user_id)
        if u is None:
            return None
        u.status = "active"
        a = self._auth.setdefault(user_id, UserAuth(id=user_id, email=u.email, role=u.role,
                                                     status=u.status, password_hash=None))
        a.password_hash = password_hash
        a.status = "active"
        a.invite_token_hash = None
        a.invite_expires_at = None
        return u

    async def set_user_status(self, user_id: str, status: str) -> Optional[User]:
        u = self._users.get(user_id)
        if u is None:
            return None
        u.status = status
        if user_id in self._auth:
            self._auth[user_id].status = status
        return u

    async def set_user_role(self, user_id: str, role: str) -> Optional[User]:
        u = self._users.get(user_id)
        if u is None:
            return None
        u.role = role
        if user_id in self._auth:
            self._auth[user_id].role = role
        return u

    async def list_users(self) -> List[User]:
        return sorted(self._users.values(), key=lambda u: u.created_at)

    async def delete_conversation(self, conversation_id) -> None:
        self._convs.pop(conversation_id, None)
        self._items.pop(conversation_id, None)
        for rid in [r.id for r in self._responses.values()
                    if r.conversation_id == conversation_id]:
            self._responses.pop(rid, None)

    async def add_memory(self, item: MemoryItem) -> MemoryItem:
        self._memories[item.id] = item
        return item

    async def list_memories(self, user_id: str, limit: int = 50) -> List[MemoryItem]:
        mems = [m for m in self._memories.values()
                if m.user_id == user_id and m.status == "active"]
        mems.sort(key=lambda m: m.updated_at, reverse=True)
        return mems[:limit]

    async def update_memory(self, memory_id: str, text: str) -> None:
        m = self._memories.get(memory_id)
        if m is not None:
            m.text = text
            m.updated_at = _now()

    async def delete_memory(self, memory_id: str) -> None:
        self._memories.pop(memory_id, None)

    async def delete_user_memories(self, user_id: str) -> None:
        for mid in [m.id for m in self._memories.values() if m.user_id == user_id]:
            self._memories.pop(mid, None)

    async def update_conversation_summary(self, conversation_id, summary, summarized_seq) -> None:
        conv = self._convs.get(conversation_id)
        if conv is not None:
            conv.summary = summary
            conv.summarized_seq = summarized_seq
