from pai_rag.integrations.chat_store.pai.pai_chat_store import LruSimpleChatStore


def test_lru_simple_chat_store():
    session_key = "abc"
    chat_store = LruSimpleChatStore()

    msgs = chat_store.get_messages(session_key)
    assert len(msgs) == 0

    msgs = ["你好", "你好。有什么能帮到你吗?"]
    for msg in msgs:
        chat_store.add_message(session_key, msg)
    assert chat_store.get_messages(session_key) == msgs

    msgs = ["你好", "Hello", "你说什么"]
    chat_store.set_messages(session_key, msgs)
    assert chat_store.get_messages(session_key) == msgs

    msgs = [f"你好_{i}" for i in range(100)]
    assert len(msgs) == 100
    chat_store.set_messages(session_key, msgs)
    expected_msgs = msgs[-20:]
    assert chat_store.get_messages(session_key) == expected_msgs

    for i in range(0, 999):
        chat_store.add_message(f"session_{i}", f"你好{i}")
    assert len(chat_store.store) == 1000
    assert chat_store.store[session_key] == expected_msgs  # 不尝试激活session_key
    chat_store.add_message("session_1000", "你好1000")
    assert len(chat_store.store) == 1000
    assert chat_store.get_messages(session_key) == []

    chat_store.get_messages("session_0")
    chat_store.add_message("session_1001", "你好1000")
    assert chat_store.store["session_0"] == ["你好0"]
    assert chat_store.store.get("session_1") is None
