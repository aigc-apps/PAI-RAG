import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from unittest.mock import MagicMock, patch


def _make_mock_tokenizer():
    """Return a tokenizer mock that counts whitespace-split tokens."""
    tok = MagicMock()
    tok.side_effect = lambda text, **kwargs: {
        "input_ids": text.split() if text else [],
        "offset_mapping": [(i, i + 1) for i in range(len(text.split()) if text else [])],
        "attention_mask": [],
    }
    return tok


@patch("agent.budgeting.get_tokenizer", return_value=_make_mock_tokenizer())
def test_fit_returns_messages_and_keeps_short_history(mock_get_tok):
    from agent.budgeting import AgentMessageManager
    from agent.message import Message

    mgr = AgentMessageManager(context_window=110000, max_output_tokens=8000)
    msgs = [Message("system", "s"), Message("user", "hello")]
    out = mgr.fit(msgs)
    assert all(isinstance(m, Message) for m in out)
    assert out[-1].content == "hello"
