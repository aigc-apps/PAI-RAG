"""Llm metric for response evaluation."""
from abc import abstractmethod
from typing import Any, Optional, Sequence

from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.prompts.mixin import PromptMixin, PromptMixinType


class LlmMetric(PromptMixin):
    """
    Llm Metric.
    """

    metric_name: str = "base"

    def __init__(
        self,
        llm: Optional[LLM] = None,
        raise_error: bool = False,
    ) -> None:
        """Init params."""
        self._llm = llm
        self._raise_error = raise_error

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
        reference_answer: str | None = None,
        contexts: Sequence[str] | None = None,
        response_answer: str | None = None,
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
