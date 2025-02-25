from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel
from loguru import logger

from llama_index.core.llms.llm import LLM
from llama_index.core import Settings
from llama_index.core import BasePromptTemplate
from llama_index.core.schema import QueryBundle
from llama_index.core.program import LLMTextCompletionProgram

from pai_rag.integrations.data_analysis.text2sql.utils.prompts import (
    DEFAULT_KEYWORD_EXTRACTION_PROMPT,
)


class QueryProcessor(ABC):
    """自然语言查询处理接口"""

    @abstractmethod
    def process(self, nl_query: QueryBundle, hint: str = None):
        pass

    @abstractmethod
    async def aprocess(self, nl_query: QueryBundle, hint: str = None):
        pass


class KeywordExtractor(QueryProcessor):
    def __init__(
        self,
        llm: Optional[LLM] = None,
        keyword_extraction_prompt: Optional[BasePromptTemplate] = None,
    ) -> None:
        self._llm = llm or Settings.llm
        self._keyword_extraction_prompt = (
            keyword_extraction_prompt or DEFAULT_KEYWORD_EXTRACTION_PROMPT
        )

    def process(self, nl_query: QueryBundle, hint: str = None) -> List[str]:
        # for llms that support FunctionCallingProgram
        # keyword_list_obj = self._llm.structured_predict(
        #     output_cls=KeywordList,
        #     prompt=self._keyword_extraction_prompt,
        #     llm_kwargs={
        #         "tool_choice": {"type": "function", "function": {"name": "KeywordList"}}
        #     },
        #     query_str=nl_query.query_str,
        #     fewshot_examples="",
        #     hint=hint,
        # )
        # for llms that does not support FunctionCallingProgram
        textCompletion = LLMTextCompletionProgram.from_defaults(
            output_cls=KeywordList,
            llm=self._llm,
            prompt=self._keyword_extraction_prompt,
        )
        keyword_list_obj = textCompletion(
            query_str=nl_query.query_str, fewshot_examples="", hint=hint
        )

        keywords = keyword_list_obj.Keywords
        # later check if parser needed
        # keywords = parse(self, keywords)
        logger.info(
            f"keyword_list: {keywords} extracted. nl_query: {nl_query.query_str}."
        )

        return keywords

    async def aprocess(self, nl_query: QueryBundle, hint: str = None) -> List[str]:
        # for llms that support FunctionCallingProgram
        # keyword_list_obj = await self._llm.astructured_predict(
        #     output_cls=KeywordList,
        #     prompt=self._keyword_extraction_prompt,
        #     llm_kwargs={
        #         "tool_choice": {"type": "function", "function": {"name": "KeywordList"}}
        #     },
        #     query_str=nl_query.query_str,
        #     fewshot_examples="",
        #     hint=hint,
        # )

        # for llms that do not support FunctionCallingProgram
        textCompletion = LLMTextCompletionProgram.from_defaults(
            output_cls=KeywordList,
            llm=self._llm,
            prompt=self._keyword_extraction_prompt,
        )
        keyword_list_obj = await textCompletion.acall(
            query_str=nl_query.query_str, fewshot_examples="", hint=hint
        )

        keywords = keyword_list_obj.Keywords
        # # later check if parser needed
        # # keywords = parse(self, keywords)
        logger.info(
            f"keyword_list: {keywords} extracted. nl_query: {nl_query.query_str}."
        )

        return keywords


class KeywordList(BaseModel):
    """Data model for KeywordList."""

    Keywords: List[str]
