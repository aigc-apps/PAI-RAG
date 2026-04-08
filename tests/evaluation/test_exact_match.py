import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
from evaluation.evaluator.exact_match_evaluator import ExactMatchEvaluator


class TestExactMatchEvaluator:
    @pytest.fixture
    def evaluator(self):
        return ExactMatchEvaluator()

    async def test_exact_match(self, evaluator):
        result = await evaluator.evaluate_async(input="q", prediction="hello", reference="hello")
        assert result["score"] == 1.0
        assert result["is_match"] is True

    async def test_case_insensitive_match(self, evaluator):
        result = await evaluator.evaluate_async(input="q", prediction="Hello", reference="hello")
        assert result["score"] == 1.0

    async def test_case_sensitive_no_match(self):
        ev = ExactMatchEvaluator(case_sensitive=True)
        result = await ev.evaluate_async(input="q", prediction="Hello", reference="hello")
        assert result["score"] == 0.0

    async def test_ignore_punctuation(self, evaluator):
        result = await evaluator.evaluate_async(input="q", prediction="hello!", reference="hello")
        assert result["score"] == 1.0

    async def test_keep_punctuation(self):
        ev = ExactMatchEvaluator(ignore_punctuation=False)
        result = await ev.evaluate_async(input="q", prediction="hello!", reference="hello")
        assert result["score"] == 0.0

    async def test_normalize_spaces(self, evaluator):
        result = await evaluator.evaluate_async(input="q", prediction="hello  world", reference="hello world")
        assert result["score"] == 1.0

    async def test_no_match(self, evaluator):
        result = await evaluator.evaluate_async(input="q", prediction="foo", reference="bar")
        assert result["score"] == 0.0
        assert result["is_match"] is False

    async def test_result_has_evaluator_name(self, evaluator):
        result = await evaluator.evaluate_async(input="q", prediction="a", reference="a")
        assert result["evaluator"] == "ExactMatch"

    async def test_custom_name(self):
        ev = ExactMatchEvaluator(name="CustomEM")
        result = await ev.evaluate_async(input="q", prediction="a", reference="a")
        assert result["evaluator"] == "CustomEM"

    def test_normalize_method(self, evaluator):
        assert evaluator._normalize("  Hello, World!  ") == "hello world"
