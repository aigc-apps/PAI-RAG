from __future__ import annotations
from typing import List, Optional
from sqlalchemy import func, delete
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from app.models import Conversation as ConvRow, ConversationItem as ItemRow, ResponseRow, UserRow
from app.store.base import Conversation, Item, StoredResponse, User, _now


def _to_item(row: ItemRow) -> Item:
    return Item(id=row.id, type=row.type, role=row.role, content=row.content or {},
                response_id=row.response_id, seq=row.seq, user_id=row.user_id)


def _to_conv(row: ConvRow) -> Conversation:
    return Conversation(id=row.id, user_id=row.user_id, title=row.title,
                        last_response_id=row.last_response_id,
                        created_at=row.created_at, updated_at=row.updated_at)


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
                              role=it.role, content=it.content, response_id=it.response_id,
                              user_id=it.user_id))
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

    async def ensure_conversation(self, conversation_id, user_id, title) -> Conversation:
        async with AsyncSession(self._engine) as s:
            row = await s.get(ConvRow, conversation_id)
            if row is None:
                row = ConvRow(id=conversation_id, user_id=user_id, title=title)
                s.add(row)
                await s.commit()
                await s.refresh(row)
            return _to_conv(row)

    async def touch_conversation(self, conversation_id, last_response_id) -> None:
        async with AsyncSession(self._engine) as s:
            row = await s.get(ConvRow, conversation_id)
            if row is None:
                return
            row.last_response_id = last_response_id
            row.updated_at = _now()
            s.add(row)
            await s.commit()

    async def list_conversations(self, user_id, limit=50, offset=0) -> List[Conversation]:
        async with AsyncSession(self._engine) as s:
            stmt = select(ConvRow)
            if user_id is not None:
                stmt = stmt.where(ConvRow.user_id == user_id)
            stmt = stmt.order_by(ConvRow.updated_at.desc()).offset(offset).limit(limit)
            rows = (await s.exec(stmt)).all()
        return [_to_conv(r) for r in rows]

    async def get_conversation(self, conversation_id) -> Optional[Conversation]:
        async with AsyncSession(self._engine) as s:
            row = await s.get(ConvRow, conversation_id)
        return _to_conv(row) if row is not None else None

    async def list_conversation_responses(self, conversation_id) -> List[StoredResponse]:
        async with AsyncSession(self._engine) as s:
            rows = (await s.exec(
                select(ResponseRow).where(
                    ResponseRow.conversation_id == conversation_id))).all()
        return [StoredResponse(id=r.id, model=r.model, status=r.status,
                               conversation_id=r.conversation_id,
                               previous_response_id=r.previous_response_id,
                               usage=r.usage, error=r.error) for r in rows]

    async def ensure_user(self, user_id, display_name=None) -> User:
        async with AsyncSession(self._engine) as s:
            row = await s.get(UserRow, user_id)
            if row is None:
                row = UserRow(id=user_id, display_name=display_name)
                s.add(row)
                await s.commit()
                await s.refresh(row)
            return User(id=row.id, display_name=row.display_name,
                        created_at=row.created_at, meta=row.meta or {})

    async def get_user(self, user_id) -> Optional[User]:
        async with AsyncSession(self._engine) as s:
            row = await s.get(UserRow, user_id)
        if row is None:
            return None
        return User(id=row.id, display_name=row.display_name,
                    created_at=row.created_at, meta=row.meta or {})

    async def delete_conversation(self, conversation_id) -> None:
        async with AsyncSession(self._engine) as s:
            await s.exec(delete(ItemRow).where(ItemRow.conversation_id == conversation_id))
            await s.exec(delete(ResponseRow).where(ResponseRow.conversation_id == conversation_id))
            await s.exec(delete(ConvRow).where(ConvRow.id == conversation_id))
            await s.commit()
