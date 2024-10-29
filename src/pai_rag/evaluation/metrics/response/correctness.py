"""Correctness evaluation."""
import asyncio
from typing import Any, Optional, Sequence, Union

from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import (
    BasePromptTemplate,
    PromptTemplate,
)
from pai_rag.evaluation.metrics.response.base import LlmMetric
from llama_index.multi_modal_llms.openai import OpenAIMultiModal


DEFAULT_EVAL_TEMPLATE = PromptTemplate(
    """
    You are an expert evaluation system for a question answering chatbot.

    You are given the following information:
    - a user query, and
    - a generated answer

    You may also be given a reference answer to use for reference in your evaluation.

    Your job is to judge the relevance and correctness of the generated answer.
    Output a single score that represents a holistic evaluation.
    You must return your response in a line with only the score.
    Do not return answers in any other format.
    On a separate line provide your reasoning for the score as well.

    Follow these guidelines for scoring:
    - Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
    - If the generated answer is not relevant to the user query, \
    you should give a score of 1.
    - If the generated answer is relevant but contains mistakes, \
    you should give a score between 2 and 3.
    - If the generated answer is relevant and fully correct, \
    you should give a score between 4 and 5.

    Example Response:
    4.0
    The generated answer has the exact same metrics as the reference answer, \
        but it is not as concise.

    ## User Query
    {query}

    ## Reference Answer
    {reference_answer}

    ## Generated Answer
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
        score_threshold: float = 4.0,
    ) -> None:
        if isinstance(eval_template, str):
            eval_template = PromptTemplate(eval_template)
        else:
            eval_template = eval_template or DEFAULT_EVAL_TEMPLATE

        super().__init__(llm, raise_error, eval_template)

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
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        del kwargs  # Unused
        del contexts  # Unused

        await asyncio.sleep(sleep_time_in_seconds)

        if query is None or response is None:
            raise ValueError("query, and response must be provided")

        # raw_response = await self._llm.apredict(
        #     prompt=self._eval_template,
        #     query=query,
        #     generated_answer=response,
        #     reference_answer=reference or "(NO REFERENCE ANSWER SUPPLIED)",
        # )
        prompt_str = self._eval_template.format(
            query=query,
            generated_answer=response,
            reference_answer=reference or "(NO REFERENCE ANSWER SUPPLIED)",
        )
        if isinstance(self._llm, OpenAIMultiModal):
            raw_response = await self._llm.acomplete(
                prompt=prompt_str, image_documents=None
            )
        else:
            raw_response = await self._llm.acomplete(prompt=prompt_str)

        # Use the parser function
        return self.parse_eval_result(str(raw_response))
