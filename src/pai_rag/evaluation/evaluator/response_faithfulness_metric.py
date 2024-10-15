"""Faithfulness evaluation."""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence, Union

from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType

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


LLAMA3_8B_EVAL_TEMPLATE = PromptTemplate(
    """Please tell if a given piece of information is supported by the context.
You need to answer with either YES or NO.
Answer YES if **any part** of the context supports the information, even if most of the context is unrelated.
Answer NO if the context does not support the information at all.
Be sure to read all provided context segments carefully before making your decision.

Some examples are provided below:

Example 1:
Information: The Eiffel Tower is located in Paris.
Context: The Eiffel Tower, a symbol of French culture, stands prominently in the city of Paris.
Answer: YES

Example 2:
Information: Bananas are a type of berry.
Context: Bananas are a popular fruit enjoyed worldwide and are rich in potassium.
Answer: NO

Example 3:
Information: Cats are reptiles.
Context: Cats are domesticated felines known for their agility and companionship.
Answer: NO

Example 4:
Information: Amazon started as an online bookstore.
Context: Amazon initially launched as an online store for books but has since expanded into a global e-commerce giant
offering various products and services.
Answer: YES

Information: {query}
Context: {reference_contexts}
Answer:"""
)

TEMPLATES_CATALOG = {"llama3:8b": LLAMA3_8B_EVAL_TEMPLATE}


class FaithfulnessEvaluator:
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
        """Init params."""
        self._llm = llm
        self._raise_error = raise_error

        self._eval_template: BasePromptTemplate
        if isinstance(eval_template, str):
            self._eval_template = PromptTemplate(eval_template)
        else:
            model_name = self._llm.metadata.model_name
            self._eval_template = eval_template or TEMPLATES_CATALOG.get(
                model_name, DEFAULT_EVAL_TEMPLATE
            )

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "eval_template": self._eval_template,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "eval_template" in prompts:
            self._eval_template = prompts["eval_template"]

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
        raw_response = await self._llm.acomplete(prompt=prompt_str)
        raw_response_txt = raw_response.text.lower()
        if "yes" in raw_response_txt:
            passing = True
        else:
            passing = False
            if self._raise_error:
                raise ValueError("The response is invalid")
        score = 1.0 if passing else 0.0
        return [score, raw_response_txt]
