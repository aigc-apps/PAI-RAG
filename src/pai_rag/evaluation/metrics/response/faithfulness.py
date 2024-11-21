"""Faithfulness evaluation."""
import asyncio
from typing import Any, Optional, Sequence, Union, List
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import (
    BasePromptTemplate,
    PromptTemplate,
)
from llama_index.core.evaluation.base import EvaluationResult
from pai_rag.evaluation.metrics.response.base import LlmMetric
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls

DEFAULT_EVAL_TEMPLATE = PromptTemplate(
    """
    请告诉我一段信息是否得到上下文的支持。
    你需要回答“是”或“否”。
    如果任何上下文支持该信息，即使大部分上下文无关，也请回答“是”。
    下面提供了一些示例。\n\n
    信息：苹果派通常是双皮的。
    上下文：苹果派是一种水果派，主要填充成分是苹果。
    苹果派通常搭配鲜奶油、冰淇淋（‘苹果派à la mode’）、奶油或切达乳酪。
    它通常是双层的，馅料上方和下方都有酥皮；上层酥皮可以是实心的或是格子状（由交叉条纹编织而成）。
    答案：是
    理由：上下文明确指出“它通常是双层的”，直接支持信息“苹果派通常是双皮的”。因此，该信息得到了上下文的确认。\n\n
    信息：苹果派味道不好。
    上下文：苹果派是一种水果派，主要填充成分是苹果。
    苹果派通常搭配鲜奶油、冰淇淋（‘苹果派à la mode’）、奶油或切达乳酪。
    它通常是双层的，馅料上方和下方都有酥皮；上层酥皮可以是实心的或是格子状（由交叉条纹编织而成）。
    答案：否
    理由：上下文没有提供关于苹果派味道的任何信息。它描述了成分和搭配建议，但没有支持“苹果派味道不好”的说法。因此，该信息没有得到上下文的支持。\n
    信息：{response_str}\n
    上下文：{context_str}\n
    答案：
    理由：
    """
)

DEFAULT_MULTIMODAL_EVAL_TEMPLATE = PromptTemplate(
    """
    请告诉我一段信息是否得到上下文的支持。
    你需要回答“是”或“否”。
    如果任何上下文支持该信息，即使大部分上下文无关，也请回答“是”。
    下面提供了一些示例。\n\n
    信息：苹果派通常是双皮的。
    上下文：苹果派是一种水果派，主要填充成分是苹果。
    苹果派通常搭配鲜奶油、冰淇淋（‘苹果派à la mode’）、奶油或切达乳酪。
    它通常是双层的，馅料上方和下方都有酥皮；上层酥皮可以是实心的或是格子状（由交叉条纹编织而成）。
    答案：是
    理由：上下文明确指出“它通常是双层的”，直接支持信息“苹果派通常是双皮的”。因此，该信息得到了上下文的确认。\n\n
    信息：苹果派味道不好。
    上下文：苹果派是一种水果派，主要填充成分是苹果。
    苹果派通常搭配鲜奶油、冰淇淋（‘苹果派à la mode’）、奶油或切达乳酪。
    它通常是双层的，馅料上方和下方都有酥皮；上层酥皮可以是实心的或是格子状（由交叉条纹编织而成）。
    答案：否
    理由：上下文没有提供关于苹果派味道的任何信息。它描述了成分和搭配建议，但没有支持“苹果派味道不好”的说法。因此，该信息没有得到上下文的支持。\n
    信息：{response_str}\n
    上下文：{context_str}\n
    参考图片链接：{reference_image_url_list}\n
    答案：
    理由：
    """
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
        multimodal_eval_template: Optional[Union[str, BasePromptTemplate]] = None,
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

    def parse_eval_result(self, eval_result: str):
        raw_response_txt = eval_result.lower()
        if "yes" or "是" in raw_response_txt:
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
        reference_answer: str | None = None,
        contexts: Sequence[str] | None = None,
        response_answer: str | None = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate whether the response is faithful to the contexts."""
        del query  # Unused
        del reference_answer  # Unused
        del kwargs  # Unused

        await asyncio.sleep(sleep_time_in_seconds)

        if contexts is None or response_answer is None:
            raise ValueError("contexts and response must be provided")

        prompt_str = self._eval_template.format(
            response_str=response_answer,
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
        del query  # Unused
        del reference_answer  # Unused
        del kwargs  # Unused
        assert isinstance(
            self._llm, OpenAIMultiModal
        ), "Multi-modal LLM must be provided to understand image."

        await asyncio.sleep(sleep_time_in_seconds)

        if (
            contexts is None and reference_image_url_list is None
        ) or response_answer is None:
            raise ValueError("contexts and response must be provided")

        image_context_str = "\n\n".join(reference_image_url_list)
        text_context_str = "\n\n".join(contexts)

        prompt_str = self._multimodal_eval_template.format(
            response_str=response_answer,
            context_str=f"{text_context_str}\n\n图片链接列表: \n\n{image_context_str}\n\n",
            reference_image_url_list=reference_image_url_list or "(没有提供参考图片链接)",
        )
        image_documents = load_image_urls(reference_image_url_list)
        raw_response = await self._llm.acomplete(
            prompt=prompt_str, image_documents=image_documents
        )

        # Use the parser function
        return self.parse_eval_result(str(raw_response))
