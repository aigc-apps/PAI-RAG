# evaluator/llm_judge_evaluator.py
from typing import Dict, Any, Optional
from evaluation.evaluator.base import BaseEvaluator
from llama_index.core.llms import LLM
from evaluation.evaluator.prompts.correctness import CORRECTNESS_PROMPT

# 基于openevals的CORRECTNESS_PROMPT进行评估
class LLMJudgeEvaluator(BaseEvaluator):
    """
    基于大语言模型的评估器
    让 LLM 扮演“裁判”角色，对预测结果打分或给出评语
    """

    def __init__(
        self,
        llm: LLM,
        name: str = "LLMJudge",
        prompt_template: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ):
        super().__init__(name)
        self.llm = llm
        self.prompt_template = prompt_template or self._default_prompt()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _default_prompt(self) -> str:
        return CORRECTNESS_PROMPT

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM，返回原始响应文本"""
        try:
            content = ""
            response_gen = await self.llm.astream_complete(prompt)
            async for r in response_gen:
                content += r.delta
            return content
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {str(e)}")

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON returned by LLM"""
        import json
        import re
        try:
            pattern = r'(?s)\{.*?\}'
            match = re.search(pattern, response_text)

            if not match:
                raise ValueError("JSON object not found")

            json_str = match.group(0)
            result = json.loads(json_str)

            if not all(key in result for key in ["score", "reason", "correctness_issues"]):
                raise ValueError("JSON missing required fields")

            return {
                "score": result.get("score", 0.0),
                "reason": result.get("reason", 0.0),
                "evaluator": self.name,
            }

        except json.JSONDecodeError as e:
            return {
                    "score": 0.0,
                    "reason": f"JSON parsing failed: {str(e)}\nOriginal text: {response_text[:200]}...",
                    "evaluator": self.name,
                    "error": str(e),
                }
        except Exception as e:
            return {
                    "score": 0.0,
                    "reason": f"Extraction failed: {str(e)}",
                    "evaluator": self.name,
                    "error": str(e),
                }



    async def evaluate_async(self, input:str, prediction: str, reference: str, **kwargs) -> Dict[str, Any]:
        prompt = self.prompt_template.format(
            inputs=input,
            outputs=prediction,
            reference_outputs=reference,
        )
        llm_response = await self._call_llm(prompt)
        result = self._parse_response(llm_response)
        return result
