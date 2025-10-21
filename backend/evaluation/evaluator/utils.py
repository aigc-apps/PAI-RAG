from typing import Optional
from pydantic import BaseModel
from enum import Enum
from backend.evaluation.evaluator.base import BaseEvaluator
from evaluation.evaluator.exact_match_evaluator import ExactMatchEvaluator
from evaluation.evaluator.llm_judge_evaluator import LLMJudgeEvaluator
from backend.evaluation.evaluator.eval_sdk_agent_evaluator import EvalSdkAgentEvaluator
from llama_index.core.llms import LLM


class SupportedEvaluators(Enum):
    EXACT_MATCH = 'ExactMatch'
    LLM_JUDGE = 'LLMJudge'
    EVAL_SDK_AGENT_EVAL = 'EvalSdkAgentEval'

class EvaluatorConfig(BaseModel):
    """
    Configuration detail
    """
    name: Optional[str] = ''
    case_sensitive: Optional[bool] = False
    ignore_punctuation: Optional[bool] = False
    llm: Optional[LLM] = None

def create_evaluator(eval_config: dict, eval_llm: LLM = None) -> BaseEvaluator:
    """
    根据配置创建评估器实例
    """
    eval_type = eval_config.get("type", "")

    if eval_type == SupportedEvaluators.EXACT_MATCH:
        return ExactMatchEvaluator(
            case_sensitive=eval_config.get("case_sensitive", False),
            ignore_punctuation=eval_config.get("case_sensitive", True)
        )
    elif eval_type == SupportedEvaluators.LLM_JUDGE:
        assert eval_llm is not None, "Must provide eval llm instance"
        return LLMJudgeEvaluator(
            llm=eval_llm
        )
    elif eval_type == SupportedEvaluators.EVAL_SDK_AGENT_EVAL:
        # TODO: set ctor arguments
        return EvalSdkAgentEvaluator()
    else:
        raise ValueError(f"不支持的评估器类型: {eval_type}")
