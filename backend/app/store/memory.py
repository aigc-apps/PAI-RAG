from __future__ import annotations
from typing import Dict, List, Optional
from app.store.base import Conversation, Item, StoredResponse


class InMemoryStore:
    def __init__(self):
        self._convs: Dict[str, Conversation] = {}
        self._items: Dict[str, List[Item]] = {}
        self._responses: Dict[str, StoredResponse] = {}

    async def create_conversation(self, user_id: Optional[str] = None) -> Conversation:
        conv = Conversation(user_id=user_id)
        self._convs[conv.id] = conv
        self._items[conv.id] = []
        return conv

    async def append_items(self, conversation_id: str, items: List[Item]) -> List[Item]:
        log = self._items.setdefault(conversation_id, [])
        for it in items:
            it.seq = len(log)
            log.append(it)
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
