import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.store.memory import InMemoryStore
from app.store.base import Item
from app.summarizer import ConversationSummarizer, maybe_summarize_conversation


def _complete_returning(payload):
    async def complete(prompt):
        return payload
    return complete


def _complete_raising():
    async def complete(prompt):
        raise RuntimeError("llm down")
    return complete


def test_summarize_returns_text_and_prior_on_failure():
    s = ConversationSummarizer(_complete_returning("NEW SUMMARY"))
    out = asyncio.run(s.summarize("old", [Item(type="message", role="user", content={"text": "hi"})]))
    assert out == "NEW SUMMARY"
    s2 = ConversationSummarizer(_complete_raising())
    assert asyncio.run(s2.summarize("PRIOR", [])) == "PRIOR"


def test_maybe_summarize_below_threshold_noop():
    async def run():
        st = InMemoryStore()
        await st.ensure_conversation("c1", user_id="u1", title="t")
        await st.append_items("c1", [Item(type="message", role="user", content={"text": f"m{i}"})
                                     for i in range(5)])
        did = await maybe_summarize_conversation(st, "c1", _complete_returning("S"),
                                                 keep_recent=20, batch=20)
        assert did is False
        assert (await st.get_conversation("c1")).summary is None
    asyncio.run(run())


def test_maybe_summarize_failure_does_not_advance_cursor_or_drop_items():
    async def run():
        st = InMemoryStore()
        await st.ensure_conversation("c1", user_id="u1", title="t")
        await st.append_items("c1", [Item(type="message", role="user", content={"text": f"m{i}"})
                                     for i in range(10)])  # seq 0..9
        # summarizer LLM fails -> summarize() returns prior ("" here) -> no progress
        did = await maybe_summarize_conversation(st, "c1", _complete_raising(),
                                                 keep_recent=3, batch=3)
        assert did is False
        conv = await st.get_conversation("c1")
        # cursor NOT advanced and no summary persisted -> items remain in history
        assert conv.summarized_seq == -1 and conv.summary is None
    asyncio.run(run())


def test_maybe_summarize_folds_overflow_and_keeps_recent():
    async def run():
        st = InMemoryStore()
        await st.ensure_conversation("c1", user_id="u1", title="t")
        await st.append_items("c1", [Item(type="message", role="user", content={"text": f"m{i}"})
                                     for i in range(10)])  # seq 0..9
        did = await maybe_summarize_conversation(st, "c1", _complete_returning("SUM"),
                                                 keep_recent=3, batch=3)  # 10 > 3+3
        assert did is True
        conv = await st.get_conversation("c1")
        assert conv.summary == "SUM"
        # folded all but the last keep_recent(3): summarized_seq == 6 (items 0..6 folded, 7,8,9 kept)
        assert conv.summarized_seq == 6
    asyncio.run(run())
