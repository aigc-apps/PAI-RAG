import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from evaluation.run import run_evaluator


class TestRunEvaluator:
    async def test_run_exact_match_evaluator(self):
        eval_config = {"type": "ExactMatch"}
        result = await run_evaluator(
            input="q",
            prediction="hello",
            reference="hello",
            eval_config=eval_config,
        )
        assert result["score"] == 1.0

    async def test_run_exact_match_no_match(self):
        eval_config = {"type": "ExactMatch"}
        result = await run_evaluator(
            input="q",
            prediction="foo",
            reference="bar",
            eval_config=eval_config,
        )
        assert result["score"] == 0.0

    async def test_run_llm_judge_evaluator(self):
        mock_llm = MagicMock()
        eval_config = {"type": "LLMJudge"}
        with patch("evaluation.evaluator.llm_judge_evaluator.LLMJudgeEvaluator.evaluate_async", new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = {"score": 0.8, "reason": "good"}
            result = await run_evaluator(
                input="q",
                prediction="answer",
                reference="ref",
                eval_config=eval_config,
                eval_llm=mock_llm,
            )
            assert result["score"] == 0.8
