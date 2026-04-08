import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from pydantic import BaseModel
from common.chat.response_model import to_dict, PagedResult


class SampleModel(BaseModel):
    name: str
    value: int


class TestToDict:
    def test_base_model(self):
        model = SampleModel(name="test", value=42)
        assert to_dict(model) == {"name": "test", "value": 42}

    def test_nested_dict(self):
        model = SampleModel(name="inner", value=1)
        result = to_dict({"outer": model})
        assert result == {"outer": {"name": "inner", "value": 1}}

    def test_list_of_models(self):
        models = [SampleModel(name="a", value=1), SampleModel(name="b", value=2)]
        result = to_dict(models)
        assert result == [{"name": "a", "value": 1}, {"name": "b", "value": 2}]

    def test_primitive_passthrough(self):
        assert to_dict("hello") == "hello"
        assert to_dict(42) == 42
        assert to_dict(None) is None


class TestPagedResult:
    def test_construction(self):
        result = PagedResult(
            items=["a", "b"],
            total=10,
            pages=5,
            page=1,
            size=2,
        )
        assert result.items == ["a", "b"]
        assert result.total == 10
        assert result.pages == 5
        assert result.page == 1
        assert result.size == 2
