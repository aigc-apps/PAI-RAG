import sys
import os
import json
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))
from agent.core.events import TextDelta, RunCompleted, RunFailed, Usage, ToolResult
from api.protocol.chat_serializer import serialize_chat_stream


async def _events(*evs):
    for e in evs:
        yield e


def _collect(gen):
    async def run():
        return [json.loads(s) async for s in gen]

    return asyncio.run(run())


def test_text_deltas_become_content_chunks():
    out = _collect(
        serialize_chat_stream(
            _events(
                TextDelta(text="he"),
                TextDelta(text="llo"),
                RunCompleted(usage=Usage(input=5, output=2, total=7)),
            ),
            model="m",
        )
    )
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in out)
    assert "hello" in content


def test_usage_reaches_final_chunk():  # regression: usage must not be dropped
    out = _collect(
        serialize_chat_stream(
            _events(
                TextDelta(text="hi"),
                RunCompleted(usage=Usage(input=5, output=9, total=14)),
            ),
            model="m",
        )
    )
    stop = [c for c in out if c["choices"][0].get("finish_reason") == "stop"][0]
    assert stop["usage"]["completion_tokens"] == 9 and stop["usage"]["total_tokens"] == 14


def test_run_failed_message_is_visible():  # regression: invisible-timeout bug
    out = _collect(
        serialize_chat_stream(
            _events(RunFailed(message="模型调用超时", error_type="llm_stream_timeout")),
            model="m",
        )
    )
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in out)
    assert "模型调用超时" in text
