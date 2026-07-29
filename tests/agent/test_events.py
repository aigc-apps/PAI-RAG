import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from agent.core.events import (
    Usage, RunStarted, TextDelta, ReasoningDelta,
    ToolStarted, ToolCompleted, ToolResult, RunCompleted, RunFailed,
)


def test_each_event_has_stable_type_tag():
    assert TextDelta(text="hi").type == "text.delta"
    assert ReasoningDelta(text="r").type == "reasoning.delta"
    assert RunStarted(response_id="resp_1").type == "run.started"
    assert ToolStarted(call_id="c1", name="echo").type == "tool.started"
    assert ToolCompleted(call_id="c1", name="echo", arguments='{"x":1}').type == "tool.completed"
    assert ToolResult(call_id="c1", name="echo", ok=True, output="r").type == "tool.result"
    assert RunCompleted(usage=Usage(input=5, output=9, total=14), finish_reason="stop").type == "run.completed"
    assert RunFailed(message="timeout", error_type="llm_stream_timeout").type == "run.failed"


def test_tool_result_carries_error_when_not_ok():
    ev = ToolResult(call_id="c1", name="echo", ok=False, error="boom")
    assert ev.ok is False and ev.error == "boom" and ev.output is None


def test_usage_totals():
    u = Usage(input=5, output=9, total=14)
    assert (u.input, u.output, u.total) == (5, 9, 14)
