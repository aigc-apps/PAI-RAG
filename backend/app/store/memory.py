from __future__ import annotations
from typing import Dict, List, Optional
from app.store.base import Conversation, Item, StoredResponse, User, _now


class InMemoryStore:
    def __init__(self):
        self._convs: Dict[str, Conversation] = {}
        self._items: Dict[str, List[Item]] = {}
        self._responses: Dict[str, StoredResponse] = {}
        self._users: Dict[str, User] = {}

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

    async def delete_conversation(self, conversation_id) -> None:
        self._convs.pop(conversation_id, None)
        self._items.pop(conversation_id, None)
        for rid in [r.id for r in self._responses.values()
                    if r.conversation_id == conversation_id]:
            self._responses.pop(rid, None)
