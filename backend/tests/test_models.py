# ruff: noqa: E402
# tests/app/test_models.py
import asyncio
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sqlalchemy import DateTime
from app.db import make_engine, create_all
from app.models import (
    BackgroundJobRow,
    Conversation,
    ConversationItem,
    KnowledgeDataSourceRow,
    KnowledgeDocumentRow,
    ResponseRow,
)
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession


def test_create_all_and_insert_roundtrip():
    async def run():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        async with AsyncSession(engine) as s:
            conv = Conversation(id="conv_1")
            s.add(conv)
            s.add(ConversationItem(id="item_1", conversation_id="conv_1", seq=0,
                                   type="message", role="user", content={"text": "hi"}))
            s.add(ResponseRow(id="resp_1", conversation_id="conv_1", model="m", status="completed"))
            await s.commit()
            got = await s.get(ResponseRow, "resp_1")
            assert got.conversation_id == "conv_1" and got.status == "completed"
    asyncio.run(run())


def test_conversation_table_has_title_and_last_response_id():
    from app.models import Conversation as ConvRow
    cols = ConvRow.__table__.columns.keys()
    assert "title" in cols and "last_response_id" in cols


def test_conversation_dataclass_defaults():
    from app.store.base import Conversation
    c = Conversation()
    assert c.title is None and c.last_response_id is None
    assert c.created_at is not None and c.updated_at is not None


def test_all_datetime_columns_are_timezone_aware():
    datetime_columns = [
        column
        for table in SQLModel.metadata.tables.values()
        for column in table.columns
        if isinstance(column.type, DateTime)
    ]

    assert datetime_columns
    assert all(column.type.timezone is True for column in datetime_columns), [
        f"{column.table.name}.{column.name}"
        for column in datetime_columns
        if column.type.timezone is not True
    ]


def test_responses_request_accepts_user_id():
    from app.schemas import ResponsesRequest
    req = ResponsesRequest(model="m", input="hi", user_id="u_123")
    assert req.user_id == "u_123"


def test_offline_pipeline_model_defaults():
    job = BackgroundJobRow(id="job_1", kind="kb_sync")
    doc = KnowledgeDocumentRow(
        id="doc_1", kb_id="kb_1", title="Doc", created_by="u_1"
    )
    source = KnowledgeDataSourceRow(
        id="ds_1", kb_id="kb_1", name="Docs", created_by="u_1"
    )

    assert job.progress == {}
    assert job.heartbeat_at is None
    assert job.lease_expires_at is None
    assert job.cancel_requested_at is None
    assert source.active_job_id is None
    assert doc.search_index_status == "indexed"
    assert doc.search_index_error is None
    assert doc.search_index_attempts == 0
