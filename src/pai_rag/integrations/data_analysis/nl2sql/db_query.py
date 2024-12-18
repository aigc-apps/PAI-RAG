import os
import json
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
from pai_rag.integrations.data_analysis.nl2sql.db_utils.constants import (
    DEFAULT_DB_DESCRIPTION_PATH,
    DEFAULT_DB_HISTORY_PATH,
)
from pai_rag.integrations.index.pai.pai_vector_index import PaiVectorStoreIndex


class DBQuery:
    """
    online workflow
    """

    def __init__(
        self,
        db_config: Dict,
        sql_database: SQLDatabase,
        embed_model: BaseEmbedding,
        description_index: PaiVectorStoreIndex,
        history_index: PaiVectorStoreIndex,
        value_index: PaiVectorStoreIndex,
        llm: Optional[LLM] = None,
        db_structured_description_path: Optional[str] = None,
        db_query_history_path: Optional[str] = None,
        keyword_extraction_prompt: Optional[BasePromptTemplate] = None,
        db_schema_select_prompt: Optional[BasePromptTemplate] = None,
        text_to_sql_prompt: Optional[BasePromptTemplate] = None,
        sql_revision_prompt: Optional[BasePromptTemplate] = None,
    ) -> None:
        self._db_config = db_config
        self._sql_database = sql_database
        self._embed_model = embed_model
        self._llm = llm or Settings.llm
        self._dialect = self._sql_database.dialect
        self._tables = list(self._sql_database._usable_tables)
        db_structured_description_path = (
            db_structured_description_path or DEFAULT_DB_DESCRIPTION_PATH
        )
        if os.path.exists(db_structured_description_path):
            with open(db_structured_description_path, "r") as f:
                self._db_description_dict = json.load(f)
        else:
            raise FileNotFoundError(
                f"Please load your db info first, db_structured_description_path: {db_structured_description_path} does not exist. "
            )
        self._enable_db_history = self._db_config.get("enable_db_history", False)
        db_query_history_path = db_query_history_path or DEFAULT_DB_HISTORY_PATH
        # print(self._db_config)
        if self._enable_db_history and os.path.exists(db_query_history_path):
            with open(db_query_history_path, "r") as f:
                self._db_history_list = json.load(f)
        else:
            self._db_history_list = []
            logger.info(
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

        self._query_preprocessor = QueryPreprocessor(
            keyword_extraction_prompt=self._keyword_extraction_prompt, llm=self._llm
        )
        self._db_preretriever = DBPreRetriever(
            embed_model=self._embed_model,
            description_index=description_index,
            history_index=history_index,
            value_index=value_index,
        )
        self._db_schema_selector = DBSelector(
            db_schema_select_prompt=self._db_schema_select_prompt, llm=self._llm
        )
        self._sql_generator = SQLGenerator(
            sql_database=self._sql_database,
            text_to_sql_prompt=self._text_to_sql_prompt,
            sql_revision_prompt=self._sql_revision_prompt,
            llm=self._llm,
            embed_model=self._embed_model,
        )

    def query_pipeline(self, nl_query: QueryBundle):
        """pipeline for sql generation"""
        if isinstance(nl_query, str):
            nl_query = QueryBundle(nl_query)

        # 查询问题预处理, 可选
        if self._enable_query_preprocessor:
            keywords = self._query_preprocessor.extract_keywords(nl_query)
            logger.info(f"Extracted keywords: {keywords}")
        else:
            keywords = []

        # 上传q-sql pair
        if (self._enable_db_history) and len(self._db_history_list) != 0:
            retrieved_db_history_list = self._db_preretriever.get_retrieved_history(
                nl_query=nl_query, top_k=3, db_history_list=self._db_history_list
            )
        else:
            retrieved_db_history_list = self._db_history_list

        # pre_retrieval
        if self._enable_db_preretriever:
            retrieved_db_description_dict = (
                self._db_preretriever.get_retrieved_description(
                    nl_query,
                    keywords,
                    top_k=5,
                    db_description_dict=self._db_description_dict,
                )
            )
        else:
            retrieved_db_description_dict = self._db_description_dict

        # schema selector, 可选
        if self._enable_db_selector:
            selected_db_description_dict = self._db_schema_selector.select_schema(
                nl_query=nl_query, db_description_dict=retrieved_db_description_dict
            )
        else:
            selected_db_description_dict = retrieved_db_description_dict

        # sql generator, 必须
        response_nodes, metadata = self._sql_generator.generate_sql_nodes(
            nl_query,
            selected_db_description_dict,
            retrieved_db_history_list,
            max_retry=1,
        )
        return response_nodes, metadata["schema_description"]

    async def aquery_pipeline(self, nl_query: QueryBundle):
        """pipeline for sql generation"""
        if isinstance(nl_query, str):
            nl_query = QueryBundle(nl_query)

        # 查询问题预处理, 可选
        if self._enable_query_preprocessor:
            keywords = self._query_preprocessor.extract_keywords(nl_query)
            logger.info(f"Extracted keywords: {keywords}")
        else:
            keywords = []

        # 上传q-sql pair
        if (self._enable_db_history) and len(self._db_history_list) != 0:
            retrieved_db_history_list = (
                await self._db_preretriever.aget_retrieved_history(
                    nl_query=nl_query, top_k=3, db_history_list=self._db_history_list
                )
            )
        else:
            retrieved_db_history_list = self._db_history_list

        # pre_retrieval
        if self._enable_db_preretriever:
            retrieved_db_description_dict = (
                await (
                    self._db_preretriever.aget_retrieved_description(
                        nl_query,
                        keywords,
                        top_k=5,
                        db_description_dict=self._db_description_dict,
                    )
                )
            )
        else:
            retrieved_db_description_dict = self._db_description_dict

        # schema selector, 可选
        if self._enable_db_selector:
            selected_db_description_dict = (
                await self._db_schema_selector.aselect_schema(
                    nl_query=nl_query, db_description_dict=retrieved_db_description_dict
                )
            )
        else:
            selected_db_description_dict = retrieved_db_description_dict

        # sql generator, 必须
        response_nodes, metadata = await self._sql_generator.agenerate_sql_nodes(
            nl_query,
            selected_db_description_dict,
            retrieved_db_history_list,
            max_retry=1,
        )
        return response_nodes, metadata["schema_description"]

    @classmethod
    def from_config(
        cls,
        sql_config: SqlAnalysisConfig,
        sql_database: SQLDatabase,
        embed_model: BaseEmbedding,
        index: List[PaiVectorStoreIndex],
        llm: Optional[LLM] = None,
    ):
        db_config = {
            "enable_db_history": sql_config.enable_db_history,
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
            embed_model=embed_model,
            description_index=index[0],
            history_index=index[1],
            value_index=index[2],
            text_to_sql_prompt=nl2sql_prompt_tmpl,
            llm=llm,
        )

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes from the database."""
        return self.query_pipeline(query_bundle)

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes from the database."""
        return await self.aquery_pipeline(query_bundle)
