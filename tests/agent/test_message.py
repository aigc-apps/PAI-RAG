import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from agent.message import Message, ToolCall


def test_text_message_roundtrips_to_wire():
    m = Message(role="user", content="hi")
    assert m.to_wire() == {"role": "user", "content": "hi"}
    assert Message.from_wire({"role": "user", "content": "hi"}) == m


def test_assistant_tool_call_to_wire():
    m = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="c1", name="read", arguments='{"x":1}')],
    )
    wire = m.to_wire()
    assert wire["role"] == "assistant"
    assert wire["content"] is None
    assert wire["tool_calls"][0] == {
        "id": "c1",
        "type": "function",
        "function": {"name": "read", "arguments": '{"x":1}'},
    }
    assert Message.from_wire(wire) == m


def test_tool_result_message_to_wire():
    m = Message(role="tool", content="result text", tool_call_id="c1")
    assert m.to_wire() == {"role": "tool", "content": "result text", "tool_call_id": "c1"}


def test_list_content_preserved_for_multimodal():
    parts = [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "http://x"}},
    ]
    m = Message(role="user", content=parts)
    assert m.to_wire()["content"] == parts


from agent.message import from_thread, keep_last_rounds


def test_from_thread_plain_user_and_assistant():
    out = from_thread([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ])
    assert [m.role for m in out] == ["user", "assistant"]
    assert out[0].content == "q1"


def test_from_thread_flattens_user_content_array_text_only():
    out = from_thread([{"role": "user", "content": [
        {"type": "text", "text": "line1"}, {"type": "text", "text": "line2"}]}])
    assert out[0].content == "line1\nline2"


def test_from_thread_keeps_image_parts():
    out = from_thread([{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "http://x"}}]}])
    assert isinstance(out[0].content, list)
    assert out[0].content[0] == {"type": "text", "text": "look"}


def test_from_thread_drops_orphan_tool_message():
    out = from_thread([{"role": "tool", "content": "x", "tool_call_id": "missing"}])
    assert out == []


def test_keep_last_rounds_trims_by_user_turn():
    msgs = [Message("user", "u1"), Message("assistant", "a1"),
            Message("user", "u2"), Message("assistant", "a2")]
    assert keep_last_rounds(msgs, 1) == msgs[2:]


def test_keep_last_rounds_zero_is_noop():
    msgs = [Message("user", "u1")]
    assert keep_last_rounds(msgs, 0) == msgs
