import logging
from typing import List, Optional
from llama_index.core.llms.llm import LLM
from llama_index.core import BasePromptTemplate
from llama_index.core.schema import QueryBundle

from pai_rag.integrations.data_analysis.nl2sql.nl2sql_prompts import (
    DEFAULT_KEYWORD_EXTRACTION_PROMPT,
)


logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """
    预处理自然语言问题，目前主要考虑关键词提取，query改写待定；
    考虑是否作为db_preretriever的子功能
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        keyword_extraction_prompt: Optional[BasePromptTemplate] = None,
    ) -> None:
        self._llm = llm
        self._keyword_extraction_prompt = (
            keyword_extraction_prompt or DEFAULT_KEYWORD_EXTRACTION_PROMPT
        )

    def keyword_extraction(self, nl_query: QueryBundle) -> List[str]:
        keyword_list = self._llm.predict(
            prompt=self._keyword_extraction_prompt, nl_query=nl_query.query_str
        )
        # later check if parser needed
        # keyword_list = parse(self, keyword_list)
        return keyword_list

    def query_transformation(self, nl_query: QueryBundle) -> List[str]:
        # 改写query以及分解等
        pass
