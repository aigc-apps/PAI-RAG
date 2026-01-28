from typing import Optional
from pydantic import BaseModel
from evaluation.evaluator.base import BaseEvaluator
from evaluation.evaluator.exact_match_evaluator import ExactMatchEvaluator
from evaluation.evaluator.llm_judge_evaluator import LLMJudgeEvaluator
from evaluation.evaluator.prompts.eval_prompts import LLM_JUDGE_PROMPT
from llama_index.core.llms import LLM

class EvaluatorConfig(BaseModel):
    """
    Condition detail
    """

    name: Optional[str] = "ExactMatch"
    case_sensitive: Optional[bool] = False
    ignore_punctuation: Optional[bool] = False
    llm: Optional[LLM] = None



def create_evaluator(eval_config: dict, eval_llm: LLM = None) -> BaseEvaluator:
    """
    根据配置创建评估器实例
    """
    eval_type = eval_config.get("type", "")
    if eval_type == "ExactMatch":
        return ExactMatchEvaluator(
            case_sensitive=eval_config.get("case_sensitive", False),
            ignore_punctuation=eval_config.get("ignore_punctuation", True)
        )
    elif eval_type == "LLMJudge":
        assert eval_llm is not None, "Must provide eval llm instance"
        # 如果配置中有自定义的 prompt（非 None 且非空字符串），使用自定义的，否则使用默认的
        custom_prompt = eval_config.get("llm_judge_prompt")
        prompt_template = (custom_prompt and custom_prompt.strip()) or LLM_JUDGE_PROMPT
        return LLMJudgeEvaluator(
            llm=eval_llm,
            prompt_template=prompt_template
        )

    else:
        raise ValueError(f"不支持的评估器类型: {eval_type}")
