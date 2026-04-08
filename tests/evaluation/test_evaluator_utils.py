import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
from unittest.mock import MagicMock
from evaluation.evaluator.utils import create_evaluator
from evaluation.evaluator.exact_match_evaluator import ExactMatchEvaluator
from evaluation.evaluator.llm_judge_evaluator import LLMJudgeEvaluator


class TestCreateEvaluator:
    def test_create_exact_match(self):
        config = {"type": "ExactMatch", "case_sensitive": True, "ignore_punctuation": False}
        ev = create_evaluator(config)
        assert isinstance(ev, ExactMatchEvaluator)
        assert ev.case_sensitive is True
        assert ev.ignore_punctuation is False

    def test_create_exact_match_defaults(self):
        config = {"type": "ExactMatch"}
        ev = create_evaluator(config)
        assert isinstance(ev, ExactMatchEvaluator)
        assert ev.case_sensitive is False
        assert ev.ignore_punctuation is True

    def test_create_llm_judge(self):
        mock_llm = MagicMock()
        config = {"type": "LLMJudge"}
        ev = create_evaluator(config, eval_llm=mock_llm)
        assert isinstance(ev, LLMJudgeEvaluator)

    def test_create_llm_judge_without_llm_raises(self):
        config = {"type": "LLMJudge"}
        with pytest.raises(AssertionError):
            create_evaluator(config)

    def test_unsupported_type_raises(self):
        config = {"type": "UnknownType"}
        with pytest.raises(ValueError, match="不支持的评估器类型"):
            create_evaluator(config)

    def test_empty_type_raises(self):
        config = {}
        with pytest.raises(ValueError):
            create_evaluator(config)
