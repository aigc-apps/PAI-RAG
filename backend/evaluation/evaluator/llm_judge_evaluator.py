# evaluator/llm_judge_evaluator.py
from typing import Dict, Any, Optional
from .base import BaseEvaluator
from llama_index.core.llms import LLM

# 假设你有一个 LLM 客户端（如 OpenAI、本地模型等）
# 你可以替换为你自己的 LLM 调用逻辑
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
        return """你是一个严谨的评估专家。请根据参考答案，评估模型预测结果的质量。

【评分标准】
- 5分：完全正确，语义一致，表达清晰
- 4分：基本正确，有轻微表达差异
- 3分：部分正确，但有明显错误或遗漏
- 2分：大部分错误，仅少量正确
- 1分：完全错误或无关

【输出格式】
请严格按照以下 JSON 格式输出：
{{
  "score": 1~5之间的整数,
  "reason": "评分理由（50字以内）"
}}

参考答案：{reference}
模型预测：{prediction}
"""

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM，返回原始响应文本"""
        try:
            response = await self.llm.acomplete(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"LLM 调用失败: {str(e)}")

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON"""
        import json
        try:
            # 尝试提取 JSON 块（兼容 ```json ... ``` 格式）
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.rfind("```")
                json_str = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.rfind("```")
                json_str = response_text[start:end].strip()
            else:
                json_str = response_text.strip()

            result = json.loads(json_str)
            score = float(result.get("score", 0))
            reason = str(result.get("reason", ""))

            # 标准化 score 到 0~1
            normalized_score = min(max(score, 1), 5) / 5.0  # 1~5 → 0.2~1.0

            return {
                "score": normalized_score,
                "raw_score": score,
                "reason": reason,
                "raw_response": response_text,
                "evaluator": self.name,
            }

        except Exception as e:
            return {
                "score": 0.0,
                "reason": f"解析失败: {str(e)[:50]}",
                "raw_response": response_text,
                "evaluator": self.name,
                "error": str(e),
            }

    async def evaluate_async(self, prediction: str, reference: str, **kwargs) -> Dict[str, Any]:
        prompt = self.prompt_template.format(
            prediction=prediction,
            reference=reference,
        )

        llm_response = await self._call_llm(prompt)
        result = self._parse_response(llm_response)
        return result

    # def evaluate(self, prediction: str, reference: str, **kwargs) -> Dict[str, Any]:
    #     """同步接口，内部调用异步方法"""
    #     try:
    #         loop = asyncio.get_event_loop()
    #         if loop.is_running():
    #             import nest_asyncio
    #             nest_asyncio.apply()
    #             future = asyncio.ensure_future(self.evaluate_async(prediction, reference, **kwargs))
    #             result = asyncio.get_event_loop().run_until_complete(future)
    #         else:
    #             # 新事件循环
    #             result = asyncio.run(self.evaluate_async(prediction, reference, **kwargs))
    #         return result
    #     except Exception as e:
    #         return {
    #             "score": 0.0,
    #             "reason": f"评估失败: {str(e)}",
    #             "evaluator": self.name,
    #             "error": str(e),
    #         }
