import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from common.llm.models import TextChunk
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction


def tool_call(idx, cid, name, args):
    return ChoiceDeltaToolCall(index=idx, id=cid, type="function",
        function=ChoiceDeltaToolCallFunction(name=name, arguments=args))


class FakeLLM:
    context_window = 110000
    max_tokens = 8000

    def __init__(self, turns):
        self._turns = list(turns)

    async def astream(self, messages, tools):
        chunks = self._turns.pop(0)
        async def gen():
            for c in chunks:
                yield c
        return gen()
