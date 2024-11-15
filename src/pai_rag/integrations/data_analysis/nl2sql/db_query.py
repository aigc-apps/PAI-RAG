import os
from loguru import logger
from typing import List, Dict, Optional

from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import BasePromptTemplate, PromptTemplate
from llama_index.core import Settings
from llama_index.core.schema import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.utilities.sql_wrapper import SQLDatabase

from pai_rag.integrations.data_analysis.data_analysis_config import SqlAnalysisConfig
from pai_rag.integrations.data_analysis.nl2sql.nl2sql_prompts import (
    DEFAULT_KEYWORD_EXTRACTION_PROMPT,
    DEFAULT_DB_SCHEMA_SELECT_PROMPT,
    DEFAULT_TEXT_TO_SQL_PROMPT,
    DEFAULT_SQL_REVISION_PROMPT,
)
from pai_rag.integrations.data_analysis.nl2sql.db_preretriever import DBPreRetriever
from pai_rag.integrations.data_analysis.nl2sql.db_selector import DBSelector
from pai_rag.integrations.data_analysis.nl2sql.query_preprocessor import (
    QueryPreprocessor,
)
from pai_rag.integrations.data_analysis.nl2sql.sql_generator import SQLGenerator


DEFAULT_DB_STRUCTURED_DESCRIPTION_PATH = (
    "./localdata/data_analysis/nl2sql/db_structured_description.txt"
)
DEFAULT_DB_QUERY_HISTORY_PATH = "./localdata/data_analysis/nl2sql/db_query_history.txt"


class DBQuery:
    """
    online workflow
    """

    def __init__(
        self,
        db_config: Dict,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        sql_database: Optional[SQLDatabase] = None,  # from offline output
        db_structured_description_path: Optional[str] = None,  # from offline output
        db_query_history_path: Optional[str] = None,  # from offline output
        keyword_extraction_prompt: Optional[BasePromptTemplate] = None,
        db_schema_select_prompt: Optional[BasePromptTemplate] = None,
        text_to_sql_prompt: Optional[BasePromptTemplate] = None,
        sql_revision_prompt: Optional[BasePromptTemplate] = None,
    ) -> None:
        self._db_config = db_config
        self._llm = llm or Settings.llm
        self._embed_model = embed_model or Settings.embed_model
        self._sql_database = sql_database
        self._dialect = self._sql_database.dialect
        self._tables = list(self._sql_database._usable_tables)

        db_structured_description_path = (
            db_structured_description_path or DEFAULT_DB_STRUCTURED_DESCRIPTION_PATH
        )
        if os.path.exists(db_structured_description_path):
            with open(db_structured_description_path, "r") as f:
                self._db_description_str = f.read()
        else:
            raise FileNotFoundError(
                f"Please load your db info first, db_structured_description_path: {db_structured_description_path} does not exist. "
            )
        db_query_history_path = db_query_history_path or DEFAULT_DB_QUERY_HISTORY_PATH
        if os.path.exists(db_query_history_path):
            with open(db_query_history_path, "r") as f:
                self._db_history_str = f.read()
        else:
            self._db_history_str = ""
            logger.warning(
                f"db_query_history_path: {db_query_history_path} does not exist, will not be used."
            )

        self._keyword_extraction_prompt = (
            keyword_extraction_prompt or DEFAULT_KEYWORD_EXTRACTION_PROMPT
        )
        self._db_schema_select_prompt = (
            db_schema_select_prompt or DEFAULT_DB_SCHEMA_SELECT_PROMPT
        )
        self._text_to_sql_prompt = text_to_sql_prompt or DEFAULT_TEXT_TO_SQL_PROMPT
        self._sql_revision_prompt = sql_revision_prompt or DEFAULT_SQL_REVISION_PROMPT

        self._enable_query_preprocessor = self._db_config.get(
            "enable_query_preprocessor", False
        )
        self._enable_db_preretriever = self._db_config.get(
            "enable_db_preretriever", False
        )
        self._enable_db_selector = self._db_config.get("enable_db_selector", False)

        if self._enable_query_preprocessor:
            self._query_preprocessor = QueryPreprocessor(
                keyword_extraction_prompt=self._keyword_extraction_prompt, llm=self._llm
            )
        else:
            self._query_preprocessor = None

        if self._enable_db_preretriever:
            self._db_preretriever = DBPreRetriever()
        else:
            self._db_preretriever = None

        if self._enable_db_selector:
            self._db_schema_selector = DBSelector(
                db_schema_select_prompt=self._db_schema_select_prompt, llm=self._llm
            )
        else:
            self._db_schema_selector = None

        self._sql_generator = SQLGenerator(
            sql_database=self._sql_database,
            text_to_sql_prompt=self._text_to_sql_prompt,
            sql_revision_prompt=self._sql_revision_prompt,
            llm=self._llm,
            embed_model=self._embed_model,
        )

    def query_pipeline(self, nl_query: QueryBundle):
        """pipeline for sql generation"""

        # # 1. 查询问题预处理, 可选
        # if self._query_preprocessor:
        #     keyword_list = self._query_preprocessor.keyword_extraction()
        # else:
        #     keyword_list = None

        # 2. pre_retrieval, 可选
        if self._db_preretriever:
            retrieved_db_description_str = self._db_description_str
            # retrieved_db_description_str = self._db_preretriever.retrieve(nl_query, self._db_description_str, keyword_list)
        else:
            retrieved_db_description_str = self._db_description_str
        if self._db_history_str != "":
            retrieved_db_history_str = self._db_history_str
            # retrieved_db_history_str = self._db_preretriever.retrieve(nl_query, self._db_history_str)
        else:
            retrieved_db_history_str = self._db_history_str

        # 3. schema selector, 可选
        if self._db_schema_selector:
            selected_db_description_str = self._db_schema_selector.select_schema(
                nl_query, retrieved_db_description_str
            )
        else:
            selected_db_description_str = retrieved_db_description_str

        # 4. sql generator, 必须
        response_nodes, _ = self._sql_generator.generate_sql_candidates(
            nl_query, selected_db_description_str, retrieved_db_history_str
        )
        return response_nodes

    async def aquery_pipeline(self, nl_query: QueryBundle):
        """pipeline for sql generation"""

        # # 1. 查询问题预处理, 可选
        # if self._query_preprocessor:
        #     keyword_list = self._query_preprocessor.keyword_extraction()
        # else:
        #     keyword_list = None

        # 2. pre_retrieval, 可选
        if self._db_preretriever:
            retrieved_db_description_str = self._db_description_str
            # retrieved_db_description_str = self._db_preretriever.retrieve(nl_query, self._db_description_str, keyword_list)
        else:
            retrieved_db_description_str = self._db_description_str
        if self._db_history_str != "":
            retrieved_db_history_str = self._db_history_str
            # retrieved_db_history_str = self._db_preretriever.retrieve(nl_query, self._db_history_str)
        else:
            retrieved_db_history_str = self._db_history_str

        # 3. schema selector, 可选
        if self._db_schema_selector:
            selected_db_description_str = await self._db_schema_selector.aselect_schema(
                nl_query, retrieved_db_description_str
            )
        else:
            selected_db_description_str = retrieved_db_description_str

        # 4. sql generator, 必须
        response_nodes, _ = await self._sql_generator.agenerate_sql_candidates(
            nl_query, selected_db_description_str, retrieved_db_history_str
        )
        return response_nodes

    @classmethod
    def from_config(
        cls,
        sql_config: SqlAnalysisConfig,
        sql_database: SQLDatabase,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
    ):
        db_config = {
            "enable_query_preprocessor": sql_config.enable_query_preprocessor,
            "enable_db_preretriever": sql_config.enable_db_preretriever,
            "enable_db_selector": sql_config.enable_db_selector,
        }

        if sql_config.nl2sql_prompt:
            nl2sql_prompt_tmpl = PromptTemplate(sql_config.nl2sql_prompt)
        else:
            nl2sql_prompt_tmpl = DEFAULT_TEXT_TO_SQL_PROMPT

        return cls(
            db_config=db_config,
            sql_database=sql_database,
            text_to_sql_prompt=nl2sql_prompt_tmpl,
            llm=llm,
            embed_model=embed_model,
        )

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes from the database."""
        return self.query_pipeline(query_bundle)

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return await self.aquery_pipeline(query_bundle)
