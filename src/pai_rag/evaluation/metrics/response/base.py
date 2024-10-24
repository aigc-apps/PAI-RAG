"""Llm metric for response evaluation."""
from abc import abstractmethod
from typing import Any, Optional, Sequence, Union

from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.prompts.mixin import PromptMixin, PromptMixinType

DEFAULT_EVAL_TEMPLATE = PromptTemplate(
    "Information: {query_str}\n" "Context: {context_str}\n" "Answer: " "Reason: "
)


class LlmMetric(PromptMixin):
    """
    Llm Metric.
    """

    metric_name: str = "base"

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
            self._eval_template = eval_template or DEFAULT_EVAL_TEMPLATE

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "eval_template": self._eval_template,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "eval_template" in prompts:
            self._eval_template = prompts["eval_template"]

    @abstractmethod
    async def parse_eval_result(self, eval_result: str) -> float:
        """Parse eval_result."""
        raise NotImplementedError

    @abstractmethod
    async def aevaluate(
        self,
        query: str | None = None,
        response: str | None = None,
        contexts: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Run evaluation with query string, retrieved contexts,
        and generated response string.

        Subclasses can override this method to provide custom evaluation logic and
        take in additional arguments.
        """
        raise NotImplementedError

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}
