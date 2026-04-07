import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from agent.context import RunContext


class TestRunContext:
    def test_from_dict_none_has_default_datetime(self):
        ctx = RunContext.from_dict(None)
        assert ctx.current_datetime
        assert re.match(r"\d{4}-\d{2}-\d{2}", ctx.current_datetime)

    def test_from_dict_with_custom_datetime(self):
        ctx = RunContext.from_dict({"current_datetime": "custom-time"})
        assert ctx.current_datetime == "custom-time"

    def test_to_string_contains_title_case(self):
        ctx = RunContext.from_dict({"current_datetime": "2024-01-01"})
        result = ctx.to_string()
        assert "Current Datetime" in result
        assert "2024-01-01" in result

    def test_empty_dict_auto_fills_datetime(self):
        ctx = RunContext.from_dict({})
        assert ctx.current_datetime
        assert len(ctx.current_datetime) > 0
