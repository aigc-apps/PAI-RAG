# tests/app/test_models.py
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.db import make_engine, create_all
from app.models import Conversation, ConversationItem, ResponseRow
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


def test_responses_request_accepts_user_id():
    from app.schemas import ResponsesRequest
    req = ResponsesRequest(model="m", input="hi", user_id="u_123")
    assert req.user_id == "u_123"
