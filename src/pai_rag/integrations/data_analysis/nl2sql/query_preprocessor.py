import json
from typing import List, Optional
from pydantic import BaseModel, Field

from llama_index.core.llms.llm import LLM
from llama_index.core import Settings
from llama_index.core import BasePromptTemplate
from llama_index.core.schema import QueryBundle

from pai_rag.integrations.data_analysis.nl2sql.nl2sql_prompts import (
    DEFAULT_KEYWORD_EXTRACTION_PROMPT,
)


class QueryPreprocessor:
    """
    预处理自然语言查询，目前主要考虑关键词提取，query改写待定；
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        keyword_extraction_prompt: Optional[BasePromptTemplate] = None,
    ) -> None:
        self._llm = llm or Settings.llm
        self._keyword_extraction_prompt = (
            keyword_extraction_prompt or DEFAULT_KEYWORD_EXTRACTION_PROMPT
        )

    def extract_keywords(self, nl_query: QueryBundle) -> List[str]:
        keyword_list_obj = self._llm.structured_predict(
            output_cls=KeywordList,
            prompt=self._keyword_extraction_prompt,
            llm_kwargs={
                "tool_choice": {"type": "function", "function": {"name": "KeywordList"}}
            },
            query_str=nl_query.query_str,
            fewshot_examples="",
        )
        # text_complection = LLMTextCompletionProgram.from_defaults(
        #         output_cls=KeywordList,
        #         prompt=self._keyword_extraction_prompt,
        # )
        # keyword_list_obj = text_complection(query_str=nl_query.query_str, fewshot_examples="")

        keywords = keyword_list_obj.Keywords
        # later check if parser needed
        # keywords = parse(self, keywords)
        # logger.info(f"keyword_list: {keywords} extracted.")
        return keywords

    async def aextract_keywords(self, nl_query: QueryBundle) -> List[str]:
        keyword_list_obj = await self._llm.astructured_predict(
            output_cls=KeywordList,
            prompt=self._keyword_extraction_prompt,
            llm_kwargs={
                "tool_choice": {"type": "function", "function": {"name": "KeywordList"}}
            },
            query_str=nl_query.query_str,
            fewshot_examples="",
        )
        keywords = keyword_list_obj.Keywords
        # later check if parser needed
        # keywords = parse(self, keywords)
        # logger.info(f"keyword_list: {keywords} extracted.")
        return keywords

    def transform_query(self, nl_query: QueryBundle) -> List[str]:
        # 考虑历史对话的query改写
        pass


class KeywordList(BaseModel):
    """Data model for KeywordList."""

    Keywords: List[str]
