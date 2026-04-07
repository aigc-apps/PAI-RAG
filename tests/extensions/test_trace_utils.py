import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from pydantic import BaseModel
from extensions.trace.utils import pydantic_to_dict


class SampleModel(BaseModel):
    name: str
    value: int


class TestPydanticToDict:
    def test_base_model_to_dict(self):
        model = SampleModel(name="test", value=42)
        result = pydantic_to_dict(model)
        assert result == {"name": "test", "value": 42}

    def test_nested_dict_with_model(self):
        model = SampleModel(name="nested", value=1)
        result = pydantic_to_dict({"key": model, "plain": "text"})
        assert result == {"key": {"name": "nested", "value": 1}, "plain": "text"}

    def test_list_with_model(self):
        models = [SampleModel(name="a", value=1), SampleModel(name="b", value=2)]
        result = pydantic_to_dict(models)
        assert result == [{"name": "a", "value": 1}, {"name": "b", "value": 2}]

    def test_primitive_types_passthrough(self):
        assert pydantic_to_dict("hello") == "hello"
        assert pydantic_to_dict(42) == 42

    def test_empty_containers(self):
        assert pydantic_to_dict({}) == {}
        assert pydantic_to_dict([]) == []
