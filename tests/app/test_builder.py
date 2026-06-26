# tests/app/test_builder.py
import sys, os, asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.schemas import ResponsesRequest
from app.builder import build_context, items_to_messages
from app.store.memory import InMemoryStore
from app.store.base import Item, StoredResponse


def test_request_ignores_unknown_fields_and_parses_input():
    req = ResponsesRequest(
        model="m", input="hello", enable_agent=True, kb_ids=["k1"]
    )
    assert req.input == "hello" and req.model == "m"


def test_build_context_from_string_input():
    async def run():
        st = InMemoryStore()
        req = ResponsesRequest(
            model="m", input="hi there", instructions="be terse"
        )
        ctx, conv_id = await build_context(req, st)
        assert ctx.current_turn.role == "user"
        assert ctx.current_turn.content == "hi there"
        assert ctx.system_prompt == "be terse"
        assert ctx.history == []
        assert conv_id is not None

    asyncio.run(run())


def test_build_context_resolves_previous_response_history():
    async def run():
        st = InMemoryStore()
        conv = await st.create_conversation()
        await st.append_items(
            conv.id,
            [
                Item(
                    type="message",
                    role="user",
                    content={"text": "q1"},
                    response_id="resp_1",
                ),
                Item(
                    type="message",
                    role="assistant",
                    content={"text": "a1"},
                    response_id="resp_1",
                ),
            ],
        )
        await st.save_response(
            StoredResponse(
                id="resp_1",
                conversation_id=conv.id,
                model="m",
                status="completed",
            )
        )
        req = ResponsesRequest(
            model="m", input="q2", previous_response_id="resp_1"
        )
        ctx, conv_id = await build_context(req, st)
        assert conv_id == conv.id
        assert [m.role for m in ctx.history] == ["user", "assistant"]
        assert ctx.history[0].content == "q1"
        assert ctx.current_turn.content == "q2"

    asyncio.run(run())


def test_items_to_messages_handles_function_call_and_output():
    msgs = items_to_messages(
        [
            Item(
                type="function_call",
                content={"call_id": "c1", "name": "get", "arguments": "{}"},
            ),
            Item(
                type="function_call_output",
                content={"call_id": "c1", "output": "42"},
            ),
        ]
    )
    assert msgs[0].role == "assistant" and msgs[0].tool_calls[0].id == "c1"
    assert (
        msgs[1].role == "tool"
        and msgs[1].tool_call_id == "c1"
        and msgs[1].content == "42"
    )


def test_build_context_from_list_input_takes_last_user_text():
    async def run():
        st = InMemoryStore()
        req = ResponsesRequest(model="m", input=[
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ])
        ctx, conv_id = await build_context(req, st)
        assert ctx.current_turn.role == "user"
        assert ctx.current_turn.content == "second"
        assert conv_id is not None

    asyncio.run(run())


def test_build_context_conflicting_ids_raises_value_error():
    async def run():
        st = InMemoryStore()
        c1 = await st.create_conversation()
        c2 = await st.create_conversation()
        await st.save_response(
            StoredResponse(
                id="resp_x",
                conversation_id=c1.id,
                model="m",
                status="completed",
            )
        )
        req = ResponsesRequest(
            model="m",
            input="q",
            previous_response_id="resp_x",
            conversation=c2.id,
        )
        import pytest

        with pytest.raises(ValueError):
            await build_context(req, st)

    asyncio.run(run())
