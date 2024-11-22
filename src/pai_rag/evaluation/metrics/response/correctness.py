"""Correctness evaluation."""
import asyncio
from typing import Any, Optional, Sequence, Union, List

from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import (
    BasePromptTemplate,
    PromptTemplate,
)
from pai_rag.evaluation.metrics.response.base import LlmMetric
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls


DEFAULT_EVAL_TEMPLATE = PromptTemplate(
    """
    你是一个问答聊天机器人的专家评估系统。
    你将获得以下信息：
    - 用户查询，以及
    - 生成的回答
    你可能还会获得一个参考答案，以供评估时参考。
    你的工作是判断生成的回答的相关性和正确性。
    输出一个代表整体评估的单一分数。
    你必须以仅包含分数的一行返回你的响应。
    不要以其他格式返回答案。
    在另一行提供你给出分数的理由。
    请遵循以下评分指南：
    - 你的分数必须在1到5之间，其中1为最低，5为最高。
    - 如果生成的回答与用户查询无关， \
    你应该给出1分。
    - 如果生成的回答相关但包含错误， \
    你应该给出2到3之间的分数。
    - 如果生成的回答相关且完全正确， \
    你应该给出4到5之间的分数。
    示例响应：
    4.0
    生成的回答与参考答案具有完全相同的指标， \
        但不够简洁。
    ## 用户查询
    {query}
    ## 参考答案
    {reference_answer}
    ## 生成的回答
    {generated_answer}
    """
)


DEFAULT_MULTIMODAL_EVAL_TEMPLATE = PromptTemplate(
    """
    你是一个问答聊天机器人的专家评估系统。
    你将获得以下信息：
    - 用户查询，以及
    - 生成的回答
    - 参考答案，以及
    - 参考图片链接

    你的工作是判断生成的回答和参考图片的相关性与正确性。
    输出一个代表整体评估的单一分数。
    你必须以仅包含分数的一行返回你的响应。
    不要以其他格式返回答案。
    在另一行提供你给出分数的理由。
    请遵循以下评分指南：
    - 你的分数必须在1到5之间，其中1为最低，5为最高。
    - 如果生成的回答与用户查询无关，
    你应该给出1分。
    - 如果生成的回答相关但包含错误，
    你应该给出2到3之间的分数。
    - 如果生成的回答相关且完全正确，
    你应该给出4到5之间的分数。
    - 另外，参考图片的质量和内容也应与用户查询和生成的回答相匹配。
    对于相关图像，如果图像的质量高且与回答内容相关，可以增加评分。

    示例响应：
    4.0
    生成的回答与参考答案具有完全相同的指标，
    但图像质量不足或不够相关。

    ## 用户查询
    {query}
    ## 参考答案
    {reference_answer}
    ## 参考图片链接
    {reference_image_url_list}
    ## 生成的回答
    {generated_answer}
    """
)


class Correctness(LlmMetric):
    """Correctness evaluator.

    Evaluates the correctness of a question answering system.
    This evaluator depends on `reference` answer to be provided, in addition to the
    query string and response string.

    It outputs a score between 1 and 5, where 1 is the worst and 5 is the best,
    along with a reasoning for the score.
    Passing is defined as a score greater than or equal to the given threshold.

    Args:
        service_context (Optional[ServiceContext]): Service context.
        eval_template (Optional[Union[BasePromptTemplate, str]]):
            Template for the evaluation prompt.
        score_threshold (float): Numerical threshold for passing the evaluation,
            defaults to 4.0.
    """

    metric_name: str = "correctness"

    def __init__(
        self,
        llm: Optional[LLM] = None,
        raise_error: bool = False,
        eval_template: Optional[Union[str, BasePromptTemplate]] = None,
        multimodal_eval_template: Optional[Union[str, BasePromptTemplate]] = None,
        score_threshold: float = 4.0,
    ) -> None:
        super().__init__(llm, raise_error)
        if isinstance(eval_template, str):
            self._eval_template = PromptTemplate(eval_template)
        else:
            self._eval_template = eval_template or DEFAULT_EVAL_TEMPLATE

        if isinstance(eval_template, str):
            self._multimodal_eval_template = PromptTemplate(multimodal_eval_template)
        else:
            self._multimodal_eval_template = (
                multimodal_eval_template or DEFAULT_MULTIMODAL_EVAL_TEMPLATE
            )

        self._score_threshold = score_threshold

    def parse_eval_result(self, eval_result: str):
        if not eval_result.strip():
            # Return None or default values if the response is empty
            return None, "No response"

        score_str, reasoning_str = eval_result.split("\n", 1)

        try:
            score = float(score_str)
        except ValueError:
            score = None

        reasoning = reasoning_str.lstrip("\n")
        return [score, reasoning]

    async def aevaluate(
        self,
        query: str | None = None,
        reference_answer: str | None = None,
        contexts: Sequence[str] | None = None,
        response_answer: str | None = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        del kwargs  # Unused
        del contexts  # Unused

        await asyncio.sleep(sleep_time_in_seconds)

        if query is None or response_answer is None:
            raise ValueError("query, and response must be provided")

        prompt_str = self._eval_template.format(
            query=query,
            generated_answer=response_answer,
            reference_answer=reference_answer or "(没有提供参考答案)",
        )
        raw_response = await self._llm.acomplete(prompt=prompt_str)

        # Use the parser function
        return self.parse_eval_result(str(raw_response))

    async def aevaluate_multimodal(
        self,
        query: str | None = None,
        reference_answer: str | None = None,
        contexts: Sequence[str] | None = None,
        reference_image_url_list: Optional[List[str]] = None,
        response_answer: str | None = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        del kwargs  # Unused
        del contexts  # Unused
        assert isinstance(
            self._llm, OpenAIMultiModal
        ), "Multi-modal LLM must be provided to understand image."

        await asyncio.sleep(sleep_time_in_seconds)

        if query is None or response_answer is None:
            raise ValueError("query, and response must be provided")

        prompt_str = self._multimodal_eval_template.format(
            query=query,
            generated_answer=response_answer,
            reference_answer=reference_answer or "(没有提供参考答案)",
            reference_image_url_list=reference_image_url_list or "(没有提供参考图片链接)",
        )
        image_documents = load_image_urls(reference_image_url_list)
        raw_response = await self._llm.acomplete(
            prompt=prompt_str, image_documents=image_documents
        )

        # Use the parser function
        return self.parse_eval_result(str(raw_response))
