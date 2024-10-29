"""Faithfulness evaluation."""
import asyncio
from typing import Any, Optional, Sequence, Union
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import (
    BasePromptTemplate,
    PromptTemplate,
)
from llama_index.core.evaluation.base import EvaluationResult
from pai_rag.evaluation.metrics.response.base import LlmMetric
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

DEFAULT_EVAL_TEMPLATE = PromptTemplate(
    "Please tell if a given piece of information "
    "is supported by the context.\n"
    "You need to answer with either YES or NO.\n"
    "Answer YES if any of the context supports the information, even "
    "if most of the context is unrelated. "
    "Some examples are provided below. \n\n"
    "Information: Apple pie is generally double-crusted.\n"
    "Context: An apple pie is a fruit pie in which the principal filling "
    "ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream "
    "('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above "
    "and below the filling; the upper crust may be solid or "
    "latticed (woven of crosswise strips).\n"
    "Answer: YES\n"
    "Reason: The context explicitly states that 'It is generally double-crusted,' "
    "which directly supports the information that 'Apple pie is generally double-crusted.' "
    "Therefore, the information is confirmed by the context. \n\n"
    "Information: Apple pies tastes bad.\n"
    "Context: An apple pie is a fruit pie in which the principal filling "
    "ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream "
    "('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above "
    "and below the filling; the upper crust may be solid or "
    "latticed (woven of crosswise strips).\n"
    "Answer: NO\n"
    "Reason: The context does not provide any information regarding the taste of apple pie. "
    "It describes the ingredients and serving suggestions but does not support the claim that "
    "'apple pies taste bad.' Therefore, the information is not supported by the context. \n"
    "Information: {query_str}\n"
    "Context: {context_str}\n"
    "Answer: "
    "Reason: "
)


class Faithfulness(LlmMetric):
    """
    Faithfulness evaluator.

    Evaluates whether a response is faithful to the contexts
    (i.e. whether the response is supported by the contexts or hallucinated.)

    This evaluator only considers the response string and the list of context strings.

    Args:
        raise_error(bool): Whether to raise an error when the response is invalid.
            Defaults to False.
        eval_template(Optional[Union[str, BasePromptTemplate]]):
            The template to use for evaluation.
    """

    metric_name: str = "faithfulness"

    def __init__(
        self,
        llm: Optional[LLM] = None,
        raise_error: bool = False,
        eval_template: Optional[Union[str, BasePromptTemplate]] = None,
    ) -> None:
        if isinstance(eval_template, str):
            eval_template = PromptTemplate(eval_template)
        else:
            eval_template = eval_template or DEFAULT_EVAL_TEMPLATE

        super().__init__(llm, raise_error, eval_template)

    def parse_eval_result(self, eval_result: str):
        raw_response_txt = eval_result.lower()
        if "yes" in raw_response_txt:
            passing = True
        else:
            passing = False
            if self._raise_error:
                raise ValueError("The response is invalid")
        score = 1.0 if passing else 0.0
        reasoning = raw_response_txt
        return [score, reasoning]

    async def aevaluate(
        self,
        query: str | None = None,
        response: str | None = None,
        contexts: Sequence[str] | None = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate whether the response is faithful to the contexts."""
        del kwargs  # Unused

        await asyncio.sleep(sleep_time_in_seconds)

        if contexts is None or response is None:
            raise ValueError("contexts and response must be provided")

        prompt_str = self._eval_template.format(
            query_str=query,
            context_str="\n".join(contexts),
        )
        if isinstance(self._llm, OpenAIMultiModal):
            raw_response = await self._llm.acomplete(
                prompt=prompt_str, image_documents=None
            )
        else:
            raw_response = await self._llm.acomplete(prompt=prompt_str)

        # Use the parser function
        return self.parse_eval_result(str(raw_response))
