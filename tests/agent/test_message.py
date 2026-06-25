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
