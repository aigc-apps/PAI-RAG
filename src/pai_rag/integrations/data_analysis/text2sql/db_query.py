import os
import json
from loguru import logger
from typing import List, Optional

from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import BasePromptTemplate
from llama_index.core import Settings
from llama_index.core.schema import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.utilities.sql_wrapper import SQLDatabase

from pai_rag.integrations.data_analysis.data_analysis_config import SqlAnalysisConfig
from pai_rag.integrations.data_analysis.text2sql.utils.prompts import (
    DEFAULT_KEYWORD_EXTRACTION_PROMPT,
    DEFAULT_DB_SCHEMA_SELECT_PROMPT,
    DEFAULT_TEXT_TO_SQL_PROMPT,
    DEFAULT_SQL_REVISION_PROMPT,
)
from pai_rag.integrations.data_analysis.text2sql.query_processor import KeywordExtractor
from pai_rag.integrations.data_analysis.text2sql.db_info_retriever import (
    SchemaRetriever,
    HistoryRetriever,
    ValueRetriever,
)
from pai_rag.integrations.data_analysis.text2sql.db_retriever_filter import (
    SchemaValueFilter,
    HistoryFilter,
)
from pai_rag.integrations.data_analysis.text2sql.db_info_selector import SchemaSelector
from pai_rag.integrations.data_analysis.text2sql.sql_generator import SQLNodeGenerator
from pai_rag.integrations.data_analysis.text2sql.utils.constants import (
    DEFAULT_DB_DESCRIPTION_PATH,
    DEFAULT_DB_DESCRIPTION_NAME,
)


class DBQuery:
    def __init__(
        self,
        db_config: SqlAnalysisConfig,
        sql_database: SQLDatabase,
        embed_model: BaseEmbedding,
        schema_retriever: Optional[SchemaRetriever] = None,
        history_retriever: Optional[HistoryRetriever] = None,
        value_retriever: Optional[ValueRetriever] = None,
        llm: Optional[LLM] = None,
        keyword_extraction_prompt: Optional[BasePromptTemplate] = None,
        db_schema_select_prompt: Optional[BasePromptTemplate] = None,
        text_to_sql_prompt: Optional[BasePromptTemplate] = None,
        sql_revision_prompt: Optional[BasePromptTemplate] = None,
    ):
        self._db_name = db_config.database
        self._sql_database = sql_database
        self._embed_model = embed_model
        self._llm = llm or Settings.llm
        self._schema_retriever = schema_retriever
        self._history_retriever = history_retriever
        self._value_retriever = value_retriever

        db_structured_description_path = os.path.join(
            DEFAULT_DB_DESCRIPTION_PATH,
            f"{self._db_name}_{DEFAULT_DB_DESCRIPTION_NAME}",
        )
        if os.path.exists(db_structured_description_path):
            with open(db_structured_description_path, "r") as f:
                self._db_description_dict = json.load(f)
        else:
            raise FileNotFoundError(
                f"Please load your db info first, {db_structured_description_path} does not exist. "
            )
        self._enable_db_history = db_config.enable_db_history
        # db_query_history_path = os.path.join(
        #     DEFAULT_DB_HISTORY_PATH, f"{self._db_name}_{DEFAULT_DB_HISTORY_NAME}"
        # )
        # # if self._enable_db_history and os.path.exists(db_query_history_path):
        # #     with open(db_query_history_path, "r") as f:
        # #         self._db_history_list = json.load(f)
        # # else:
        # #     self._db_history_list = []
        # #     logger.info("db_query_history is not enabled and will not be used.")

        self._keyword_extraction_prompt = (
            keyword_extraction_prompt or DEFAULT_KEYWORD_EXTRACTION_PROMPT
        )
        self._db_schema_select_prompt = (
            db_schema_select_prompt or DEFAULT_DB_SCHEMA_SELECT_PROMPT
        )
        self._text_to_sql_prompt = text_to_sql_prompt or DEFAULT_TEXT_TO_SQL_PROMPT
        self._sql_revision_prompt = sql_revision_prompt or DEFAULT_SQL_REVISION_PROMPT

        self._enable_query_preprocessor = db_config.enable_query_preprocessor
        self._enable_db_preretriever = db_config.enable_db_preretriever
        self._enable_db_selector = db_config.enable_db_selector

        self._keyword_extractor = KeywordExtractor(
            llm=self._llm,
            keyword_extraction_prompt=self._keyword_extraction_prompt,
        )
        self._schema_value_filter = SchemaValueFilter()
        self._history_filter = HistoryFilter()
        self._db_schema_selector = SchemaSelector(
            llm=self._llm, db_schema_select_prompt=self._db_schema_select_prompt
        )
        self._sql_generator = SQLNodeGenerator(
            sql_database=self._sql_database,
            llm=self._llm,
            embed_model=self._embed_model,
            text_to_sql_prompt=self._text_to_sql_prompt,
            sql_revision_prompt=self._sql_revision_prompt,
        )

    def query_pipeline(self, nl_query: QueryBundle, hint: str = None):
        if isinstance(nl_query, str):
            nl_query = QueryBundle(nl_query)

        # 查询问题预处理, 可选
        if self._enable_query_preprocessor:
            keywords = self._keyword_extractor.process(nl_query, hint)
        else:
            keywords = []

        # 筛选q-sql pair, 可选
        if self._enable_db_history:
            # history info retrieval
            retrieved_history_nodes = self._history_retriever.retrieve_nodes(nl_query)
            logger.info(
                f"History nodes retrieved with number {len(retrieved_history_nodes)}"
            )
            # history filter
            retrieved_history_list = self._history_filter.filter(
                retrieved_history_nodes
            )
        else:
            retrieved_history_list = []

        # pre_retrieval, 可选
        if self._enable_db_preretriever:
            # schema info retrieval
            retrieved_description_nodes_from_query = (
                self._schema_retriever.retrieve_nodes(nl_query)
            )
            if hint:
                retrieved_description_nodes_from_hint = (
                    self._schema_retriever.retrieve_nodes(hint)
                )
            else:
                retrieved_description_nodes_from_hint = []
            # value info retrieval
            if keywords:
                retrieved_value_nodes = self._value_retriever.retrieve_nodes(keywords)
            else:
                retrieved_value_nodes = []
            # schema+value filter
            retrieved_description_dict = self._schema_value_filter.filter(
                self._db_description_dict,
                retrieved_description_nodes_from_query,
                retrieved_description_nodes_from_hint,
                retrieved_value_nodes,
            )
        else:
            retrieved_description_dict = self._db_description_dict

        # schema selector, 可选
        if self._enable_db_selector:
            selected_description_dict = self._db_schema_selector.select(
                query=nl_query, db_info=retrieved_description_dict, hint=hint
            )
        else:
            selected_description_dict = retrieved_description_dict

        # sql generator, 必须
        response_node, metadata = self._sql_generator.generate_sql_node(
            nl_query,
            selected_description_dict,
            retrieved_history_list,
            max_retry=1,
            hint=hint,
        )
        return response_node, metadata["schema_description"]

    async def aquery_pipeline(self, nl_query: QueryBundle, hint: str = None):
        if isinstance(nl_query, str):
            nl_query = QueryBundle(nl_query)

        # 查询问题预处理, 可选
        if self._enable_query_preprocessor:
            keywords = await self._keyword_extractor.aprocess(nl_query, hint)
        else:
            keywords = []

        # 筛选q-sql pair, 可选
        # if (self._enable_db_history) and len(self._db_history_list) != 0:
        if self._enable_db_history:
            # history info retrieval
            retrieved_history_nodes = await self._history_retriever.aretrieve_nodes(
                nl_query
            )
            logger.info(
                f"History nodes retrieved with number {len(retrieved_history_nodes)}"
            )
            # history filter
            retrieved_history_list = self._history_filter.filter(
                retrieved_history_nodes
            )
        else:
            # retrieved_history_list = self._db_history_list
            retrieved_history_list = []

        # pre_retrieval, 可选
        if self._enable_db_preretriever:
            # schema info retrieval
            retrieved_description_nodes_from_query = (
                await self._schema_retriever.aretrieve_nodes(nl_query)
            )
            if hint:
                retrieved_description_nodes_from_hint = (
                    await self._schema_retriever.aretrieve_nodes(hint)
                )
            else:
                retrieved_description_nodes_from_hint = []
            # value info retrieval
            if keywords:
                retrieved_value_nodes = await self._value_retriever.aretrieve_nodes(
                    keywords
                )
            else:
                retrieved_value_nodes = []
            # schema+value filter
            retrieved_description_dict = self._schema_value_filter.filter(
                self._db_description_dict,
                retrieved_description_nodes_from_query,
                retrieved_description_nodes_from_hint,
                retrieved_value_nodes,
            )
        else:
            retrieved_description_dict = self._db_description_dict

        # schema selector, 可选
        if self._enable_db_selector:
            selected_description_dict = await self._db_schema_selector.aselect(
                query=nl_query, db_info=retrieved_description_dict, hint=hint
            )
        else:
            selected_description_dict = retrieved_description_dict

        # sql generator, 必须
        response_node, metadata = await self._sql_generator.agenerate_sql_node(
            nl_query,
            selected_description_dict,
            retrieved_history_list,
            max_retry=1,
            hint=hint,
        )
        return response_node, metadata["schema_description"]

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve sql nodes from the database."""
        return self.query_pipeline(query_bundle)

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve sql nodes from the database."""
        return await self.aquery_pipeline(query_bundle)
