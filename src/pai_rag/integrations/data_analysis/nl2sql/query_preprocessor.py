import json
from typing import List, Optional
from pydantic.v1 import BaseModel, Field

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
        sllm = self._llm.as_structured_llm(output_cls=KeywordList)
        keyword_list = sllm.predict(
            prompt=self._keyword_extraction_prompt,
            query_str=nl_query.query_str,
            fewshot_examples="",
        )
        keywords = json.loads(keyword_list)["Keywords"]
        # later check if parser needed
        # keywords = parse(self, keywords)
        # logger.info(f"keyword_list: {keywords} extracted.")
        return keywords

    async def aextract_keywords(self, nl_query: QueryBundle) -> List[str]:
        sllm = self._llm.as_structured_llm(output_cls=KeywordList)
        keyword_list = await sllm.predict(
            prompt=self._keyword_extraction_prompt,
            query_str=nl_query.query_str,
            fewshot_examples="",
        )
        keywords = json.loads(keyword_list)["Keywords"]
        # later check if parser needed
        # keywords = parse(self, keywords)
        # logger.info(f"keyword_list: {keywords} extracted.")
        return keywords

    def transform_query(self, nl_query: QueryBundle) -> List[str]:
        # 考虑历史对话的query改写
        pass


class KeywordList(BaseModel):
    """Data model for KeywordList."""

    Keywords: List[str] = Field(description="从查询问题中提取的关键词、关键短语和命名实体列表")
