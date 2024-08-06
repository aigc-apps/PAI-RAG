from typing import Any, List, Sequence, Optional, Dict, cast
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.types import BaseOutputParser
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.base.base_selector import BaseSelector
from llama_index.core.schema import QueryBundle
from llama_index.core.service_context_elements.llm_predictor import (
    LLMPredictorType,
)
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.output_parsers.base import StructuredOutput
from pai_rag.modules.intentdetection.output_parser import Answer, SelectionOutputParser
from llama_index.core.selectors.llm_selectors import _build_choices_text

DEFAULT_LLM_SINGLE_DETECTOR_PROMPT_TMPL = (
    "Some intents are given below. It is provided in a numbered list "
    "(1 to {num_intent}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{intent_list}"
    "\n---------------------\n"
    "Using only the intents above and not prior knowledge, return "
    "the intent that is most relevant to the question: '{query_str}'\n"
)


class SelectorResult(BaseModel):
    """A single selection of a choice."""

    index: int
    intent: str


class MultiSelectorResult(BaseModel):
    """A single selection of a choice."""

    results: List[SelectorResult]


def _structured_output_to_detector_result(
    output: Any, choices: Sequence[ToolMetadata]
) -> SelectorResult:
    """Convert structured output to selector result."""
    structured_output = cast(StructuredOutput, output)
    answers = cast(List[Answer], structured_output.parsed_output)

    # adjust for zero indexing
    selections = [
        SelectorResult(index=answer.index - 1, intent=choices[answer.index - 1].name)
        for answer in answers
    ]
    return selections[0]


class LLMSingleDetector(BaseSelector):
    """LLM single selector.

    LLM-based selector that chooses one out of many options.

    Args:
        LLM (LLM): An LLM.
        prompt (SingleSelectPrompt): A LLM prompt for selecting one out of many options.
    """

    def __init__(
        self,
        llm: Optional[LLMPredictorType] = None,
        prompt_template_str: Optional[str] = None,
        output_parser: Optional[BaseOutputParser] = None,
        choices: Optional[Sequence[ToolMetadata]] = None,
    ) -> None:
        self._llm = llm
        prompt_template_str = (
            prompt_template_str or DEFAULT_LLM_SINGLE_DETECTOR_PROMPT_TMPL
        )
        self._output_parser = output_parser or SelectionOutputParser()
        prompt = PromptTemplate(
            template=prompt_template_str,
            output_parser=self._output_parser,
            prompt_type=PromptType.CUSTOM,
        )
        self._prompt = prompt
        self._choices = choices

    @classmethod
    def from_defaults(
        cls,
        llm: Optional[LLMPredictorType] = None,
        prompt_template_str: Optional[str] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> "LLMSingleDetector":
        # optionally initialize defaults
        llm = llm
        prompt_template_str = (
            prompt_template_str or DEFAULT_LLM_SINGLE_DETECTOR_PROMPT_TMPL
        )
        output_parser = output_parser or SelectionOutputParser()

        # construct prompt
        prompt = PromptTemplate(
            template=prompt_template_str,
            output_parser=output_parser,
            prompt_type=PromptType.CUSTOM,
        )
        return cls(llm, prompt)

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {"prompt": self._prompt}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "prompt" in prompts:
            self._prompt = prompts["prompt"]

    def _select(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        # prepare input
        intent_text = _build_choices_text(choices)

        # predict
        prediction = self._llm.predict(
            prompt=self._prompt,
            num_intent=len(choices),
            intent_list=intent_text,
            query_str=query.query_str,
        )

        # parse output
        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        return _structured_output_to_detector_result(parse, choices)

    async def _aselect(
        self, choices: Sequence[ToolMetadata], query: QueryBundle
    ) -> SelectorResult:
        # prepare input
        intent_text = _build_choices_text(choices)

        # predict
        prediction = await self._llm.apredict(
            prompt=self._prompt,
            num_intent=len(choices),
            intent_list=intent_text,
            query_str=query.query_str,
        )

        # parse output
        assert self._prompt.output_parser is not None
        parse = self._prompt.output_parser.parse(prediction)
        return _structured_output_to_detector_result(parse, choices)
