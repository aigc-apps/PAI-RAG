from __future__ import annotations
from typing import List, Optional
from sqlalchemy import func, delete
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from app.models import Conversation as ConvRow, ConversationItem as ItemRow, ResponseRow
from app.store.base import Conversation, Item, StoredResponse


def _to_item(row: ItemRow) -> Item:
    return Item(id=row.id, type=row.type, role=row.role, content=row.content or {},
                response_id=row.response_id, seq=row.seq)


class SqlStore:
    def __init__(self, engine):
        self._engine = engine

    async def create_conversation(self, user_id: Optional[str] = None) -> Conversation:
        conv = Conversation(user_id=user_id)
        async with AsyncSession(self._engine) as s:
            s.add(ConvRow(id=conv.id, user_id=user_id))
            await s.commit()
        return conv

    async def append_items(self, conversation_id: str, items: List[Item]) -> List[Item]:
        async with AsyncSession(self._engine) as s:
            base = (await s.exec(
                select(func.coalesce(func.max(ItemRow.seq), -1)).where(
                    ItemRow.conversation_id == conversation_id))).one()
            n = int(base) + 1
            for it in items:
                it.seq = n
                s.add(ItemRow(id=it.id, conversation_id=conversation_id, seq=n, type=it.type,
                              role=it.role, content=it.content, response_id=it.response_id))
                n += 1
            await s.commit()
        return items

    async def get_conversation_items(self, conversation_id: str) -> List[Item]:
        async with AsyncSession(self._engine) as s:
            rows = (await s.exec(
                select(ItemRow).where(
                    ItemRow.conversation_id == conversation_id).order_by(ItemRow.seq))).all()
        return [_to_item(r) for r in rows]

    async def save_response(self, response: StoredResponse) -> StoredResponse:
        async with AsyncSession(self._engine) as s:
            s.add(ResponseRow(id=response.id, conversation_id=response.conversation_id,
                              previous_response_id=response.previous_response_id,
                              model=response.model, status=response.status,
                              usage=response.usage, error=response.error))
            await s.commit()
        return response

    async def get_response(self, response_id: str) -> Optional[StoredResponse]:
        async with AsyncSession(self._engine) as s:
            row = await s.get(ResponseRow, response_id)
        if row is None:
            return None
        return StoredResponse(id=row.id, model=row.model, status=row.status,
                              conversation_id=row.conversation_id,
                              previous_response_id=row.previous_response_id,
                              usage=row.usage, error=row.error)

    async def delete_response(self, response_id: str) -> None:
        async with AsyncSession(self._engine) as s:
            await s.exec(delete(ResponseRow).where(ResponseRow.id == response_id))
            await s.commit()

    async def resolve_history(self, previous_response_id: Optional[str],
                              conversation: Optional[str]) -> List[Item]:
        conv_id = conversation
        if previous_response_id:
            resp = await self.get_response(previous_response_id)
            if resp is None:
                return []
            if conversation and resp.conversation_id != conversation:
                raise ValueError("previous_response_id does not belong to conversation")
            conv_id = resp.conversation_id
        if not conv_id:
            return []
        return await self.get_conversation_items(conv_id)
