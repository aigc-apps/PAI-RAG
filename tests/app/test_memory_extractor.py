import sys, os, asyncio, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.store.memory import InMemoryStore
from app.store.base import MemoryItem
from app.memory import MemoryExtractor, apply_memory_ops, update_user_memory


def _complete_returning(payload):
    async def complete(prompt):
        return payload
    return complete


def test_extract_parses_ops_and_tolerates_fences():
    ex = MemoryExtractor(_complete_returning(
        '```json\n[{"op":"ADD","text":"likes tea"},{"op":"NOOP"}]\n```'))
    ops = asyncio.run(ex.extract("I like tea", "Noted.", []))
    assert ops[0]["op"] == "ADD" and ops[0]["text"] == "likes tea"


def test_extract_bad_json_returns_empty():
    ex = MemoryExtractor(_complete_returning("not json at all"))
    assert asyncio.run(ex.extract("x", "y", [])) == []


def test_apply_ops_add_update_delete():
    async def run():
        st = InMemoryStore()
        existing = await st.add_memory(MemoryItem(user_id="u1", text="old"))
        await apply_memory_ops(st, "u1", [
            {"op": "ADD", "text": "new fact"},
            {"op": "UPDATE", "target_id": existing.id, "text": "updated"},
            {"op": "DELETE", "target_id": "missing"},  # no-op safe
            {"op": "NOOP"},
        ], source_response_id="resp_1")
        texts = {m.text for m in await st.list_memories("u1")}
        assert texts == {"new fact", "updated"}
    asyncio.run(run())


def test_update_user_memory_end_to_end_with_fake_complete():
    async def run():
        st = InMemoryStore()
        complete = _complete_returning('[{"op":"ADD","text":"works at Acme"}]')
        await update_user_memory(st, "u1", "I work at Acme", "Got it.", complete, "resp_9")
        mems = await st.list_memories("u1")
        assert [m.text for m in mems] == ["works at Acme"]
        assert mems[0].source_response_id == "resp_9"
    asyncio.run(run())


def test_update_user_memory_swallows_errors():
    async def run():
        st = InMemoryStore()
        async def boom(prompt):
            raise RuntimeError("llm down")
        # must not raise
        await update_user_memory(st, "u1", "x", "y", boom)
        assert await st.list_memories("u1") == []
    asyncio.run(run())


def test_extract_tolerates_braces_in_user_content():
    # user content with {curly} braces must NOT break prompt building (no str.format)
    ex = MemoryExtractor(_complete_returning('[{"op":"ADD","text":"writes Python dicts"}]'))
    ops = asyncio.run(ex.extract("I write {config} = {} dicts", "Noted {x}.", []))
    assert ops and ops[0]["op"] == "ADD"


def test_extract_non_list_json_returns_empty():
    ex = MemoryExtractor(_complete_returning('{"op":"ADD","text":"x"}'))  # object, not array
    assert asyncio.run(ex.extract("a", "b", [])) == []


def test_make_complete_collects_deltas():
    from app.memory import make_complete
    from common.llm.models import TextChunk

    class _LLM:
        async def astream(self, messages, tools=None, **kwargs):
            async def gen():
                yield TextChunk(delta="hel", usage=None)
                yield TextChunk(delta="lo", usage=None)
            return gen()
    complete = make_complete(_LLM())
    assert asyncio.run(complete("prompt")) == "hello"
