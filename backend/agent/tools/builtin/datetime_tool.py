from __future__ import annotations
from agent.tools.base import Tool
from utils.time_utils import get_current_time_str


def make_current_datetime_tool() -> Tool:
    async def fn() -> str:
        return get_current_time_str()

    return Tool(
        name="current_datetime",
        description="Return the current local date and time (use for 'what day/time is it' and time-relative reasoning).",
        parameters={"type": "object", "properties": {}, "required": []},
        fn=fn,
    )
