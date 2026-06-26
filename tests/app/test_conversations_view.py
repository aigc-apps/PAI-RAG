import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.store.base import Item, StoredResponse
from app.conversations_view import group_conversation_messages


def _items(seq_triples):
    # seq_triples: list of (type, role, content, response_id)
    out = []
    for i, (t, role, content, rid) in enumerate(seq_triples):
        out.append(Item(type=t, role=role, content=content, response_id=rid, seq=i))
    return out


def test_single_text_turn():
    items = _items([
        ("message", "user", {"text": "hi"}, "resp_1"),
        ("message", "assistant", {"text": "hello"}, "resp_1"),
    ])
    resps = [StoredResponse(id="resp_1", model="m", status="completed",
                            conversation_id="c", previous_response_id=None)]
    msgs = group_conversation_messages(items, resps)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0] == {"role": "user", "text": "hi", "response_id": "resp_1"}
    a = msgs[1]
    assert a["text"] == "hello" and a["reasoning"] is None
    assert a["status"] == "completed" and a["previous_response_id"] is None
    assert a["response_id"] == "resp_1"


def test_reasoning_surfaced_on_assistant():
    items = _items([
        ("message", "user", {"text": "q"}, "resp_1"),
        ("reasoning", None, {"text": "let me think"}, "resp_1"),
        ("message", "assistant", {"text": "a"}, "resp_1"),
    ])
    resps = [StoredResponse(id="resp_1", model="m", status="completed", conversation_id="c")]
    msgs = group_conversation_messages(items, resps)
    assert msgs[1]["reasoning"] == "let me think"


def test_multi_turn_chronological_with_previous_response_id():
    items = _items([
        ("message", "user", {"text": "q1"}, "resp_1"),
        ("message", "assistant", {"text": "a1"}, "resp_1"),
        ("message", "user", {"text": "q2"}, "resp_2"),
        ("message", "assistant", {"text": "a2"}, "resp_2"),
    ])
    resps = [
        StoredResponse(id="resp_1", model="m", status="completed", conversation_id="c"),
        StoredResponse(id="resp_2", model="m", status="completed", conversation_id="c",
                       previous_response_id="resp_1"),
    ]
    msgs = group_conversation_messages(items, resps)
    assert [m.get("text") for m in msgs] == ["q1", "a1", "q2", "a2"]
    assert msgs[3]["previous_response_id"] == "resp_1"


def test_failed_turn_renders_empty_assistant_with_failed_status():
    items = _items([
        ("message", "user", {"text": "boom?"}, "resp_1"),
        # no assistant message item on a failed turn
    ])
    resps = [StoredResponse(id="resp_1", model="m", status="failed", conversation_id="c")]
    msgs = group_conversation_messages(items, resps)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["text"] == "" and msgs[1]["status"] == "failed"


def test_function_call_items_are_skipped():
    items = _items([
        ("message", "user", {"text": "use a tool"}, "resp_1"),
        ("function_call", None, {"call_id": "c1", "name": "get", "arguments": "{}"}, "resp_1"),
        ("function_call_output", None, {"call_id": "c1", "output": "42"}, "resp_1"),
        ("message", "assistant", {"text": "done"}, "resp_1"),
    ])
    resps = [StoredResponse(id="resp_1", model="m", status="completed", conversation_id="c")]
    msgs = group_conversation_messages(items, resps)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["text"] == "done"
