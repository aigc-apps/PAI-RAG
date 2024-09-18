"""
Modification based on llama-index SQL Retriever,
 - add score=1.0 to NodeWithScore to be compatible with my_retriever_query_engine
 - add logger for Predicted SQL query & SQL query result for synthesize
 - constrain LIMIT on the generated SQL query
 - constrain time on run_query
 - modify DefaultSQLParser
"""

import logging
import re
import signal
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.instrumentation import DispatcherSpanMixin
from llama_index.core.llms.llm import LLM
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.objects.table_node_mapping import SQLTableSchema
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_TEXT_TO_SQL_PROMPT,
)
from llama_index.core.prompts.mixin import (
    PromptDictType,
    PromptMixin,
    PromptMixinType,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType, TextNode
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    embed_model_from_settings_or_context,
    llm_from_settings_or_context,
)
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from sqlalchemy import Table

logger = logging.getLogger(__name__)


def timeout_handler():
    raise TimeoutError("Query timed out")


class MySQLRetriever(BaseRetriever):
    """SQL Retriever.

    Retrieves via raw SQL statements.

    Args:
        sql_database (SQLDatabase): SQL database.
        return_raw (bool): Whether to return raw results or format results.
            Defaults to True.

    """

    def __init__(
        self,
        sql_database: SQLDatabase,
        return_raw: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._sql_database = sql_database
        self._return_raw = return_raw
        super().__init__(callback_manager)

    def _format_node_results(
        self, results: List[List[Any]], col_keys: List[str]
    ) -> List[NodeWithScore]:
        """Format node results."""
        nodes = []
        for result in results:
            # associate column keys with result tuple
            metadata = dict(zip(col_keys, result))
            # NOTE: leave text field blank for now
            text_node = TextNode(
                text="",
                metadata=metadata,
            )
            nodes.append(NodeWithScore(node=text_node, score=1.0))
        return nodes

    def _limit_check(self, sql_query: str, max_limit=100):
        limit_pattern = r"\bLIMIT\s+(\d+)(?:\s+OFFSET\s+\d+)?\b"
        match = re.search(limit_pattern, sql_query, re.IGNORECASE)

        if match:
            limit_value = int(match.group(1))
            if limit_value > max_limit:
                new_sql_query = re.sub(
                    limit_pattern,
                    f"LIMIT {max_limit}",
                    sql_query,
                    count=1,
                    flags=re.IGNORECASE,
                )
                return new_sql_query
            else:
                return sql_query
        else:
            raise ValueError("check sql query and regular expression")

    def retrieve_with_metadata(
        self, str_or_query_bundle: QueryType
    ) -> Tuple[List[NodeWithScore], Dict]:
        """Retrieve with metadata."""
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle

        # constrain LIMIT in sql_query
        if ("INSERT" in query_bundle.query_str) or ("CREATE" in query_bundle.query_str):
            raise ValueError("ONLY QUERY ALLOWED")
        if "limit" not in query_bundle.query_str.lower():
            query_bundle.query_str = query_bundle.query_str + " limit 100"
        else:
            query_bundle.query_str = self._limit_check(query_bundle.query_str)
        logger.info(f"Limited SQL query: {query_bundle.query_str}")

        # set timeout to 10s
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # start
        try:
            raw_response_str, metadata = self._sql_database.run_sql(
                query_bundle.query_str
            )
        except (TimeoutError, NotImplementedError) as error:
            logger.info("Invalid SQL or SQL Query Timed Out (>10s)")
            raise error
            # raw_response_str = "Invalid SQL or SQL Query Timed Out (>10s)"
            # metadata = {"result": {e}, "col_keys": []}
        finally:
            signal.alarm(0)  # cancel

        if self._return_raw:
            return [
                NodeWithScore(
                    node=TextNode(
                        text=raw_response_str,
                        metadata={
                            "query_code_instruction": query_bundle.query_str,
                            "query_output": str(metadata["result"]),
                            "col_keys": metadata["col_keys"],
                        },
                        excluded_embed_metadata_keys=[
                            "query_code_instruction",
                            "query_output",
                            "col_keys",
                        ],
                        excluded_llm_metadata_keys=[
                            "query_code_instruction",
                            "query_output",
                            "col_keys",
                        ],
                    ),
                    score=1.0,
                ),
            ], metadata
        else:
            # return formatted
            results = metadata["result"]
            col_keys = metadata["col_keys"]
            return self._format_node_results(results, col_keys), metadata

    async def aretrieve_with_metadata(
        self, str_or_query_bundle: QueryType
    ) -> Tuple[List[NodeWithScore], Dict]:
        return self.retrieve_with_metadata(str_or_query_bundle)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        retrieved_nodes, _ = self.retrieve_with_metadata(query_bundle)
        return retrieved_nodes


class SQLParserMode(str, Enum):
    """SQL Parser Mode."""

    DEFAULT = "default"
    PGVECTOR = "pgvector"


class BaseSQLParser(DispatcherSpanMixin, ABC):
    """Base SQL Parser."""

    @abstractmethod
    def parse_response_to_sql(self, response: str, query_bundle: QueryBundle) -> str:
        """Parse response to SQL."""


class DefaultSQLParser(BaseSQLParser):
    """Default SQL Parser."""

    def parse_response_to_sql(self, response: str, query_bundle: QueryBundle) -> str:
        """Parse response to SQL."""
        sql_query_start = response.find("SQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            # TODO: move to removeprefix after Python 3.9+
            if response.startswith("SQLQuery:"):
                response = response[len("SQLQuery:") :]
        sql_result_start = response.find("SQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start]
        return response.strip().strip("```").strip().strip(";").strip()


class MyNLSQLRetriever(BaseRetriever, PromptMixin):
    """Text-to-SQL Retriever.

    Retrieves via text.

    Args:
        sql_database (SQLDatabase): SQL database.
        text_to_sql_prompt (BasePromptTemplate): Prompt template for text-to-sql.
            Defaults to DEFAULT_TEXT_TO_SQL_PROMPT.
        context_query_kwargs (dict): Mapping from table name to context query.
            Defaults to None.
        tables (Union[List[str], List[Table]]): List of table names or Table objects.
        table_retriever (ObjectRetriever[SQLTableSchema]): Object retriever for
            SQLTableSchema objects. Defaults to None.
        context_str_prefix (str): Prefix for context string. Defaults to None.
        service_context (ServiceContext): Service context. Defaults to None.
        return_raw (bool): Whether to return plain-text dump of SQL results, or parsed into Nodes.
        handle_sql_errors (bool): Whether to handle SQL errors. Defaults to True.
        sql_only (bool) : Whether to get only sql and not the sql query result.
            Default to False.
        llm (Optional[LLM]): Language model to use.

    """

    def __init__(
        self,
        sql_database: SQLDatabase,
        text_to_sql_prompt: Optional[BasePromptTemplate] = None,
        context_query_kwargs: Optional[dict] = None,
        tables: Optional[Union[List[str], List[Table]]] = None,
        table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
        context_str_prefix: Optional[str] = None,
        sql_parser_mode: SQLParserMode = SQLParserMode.DEFAULT,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        service_context: Optional[ServiceContext] = None,
        return_raw: bool = True,
        handle_sql_errors: bool = True,
        sql_only: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._sql_retriever = MySQLRetriever(sql_database, return_raw=return_raw)
        self._sql_database = sql_database
        self._get_tables = self._load_get_tables_fn(
            sql_database, tables, context_query_kwargs, table_retriever
        )
        self._tables = tables
        self._tables = tables
        self._context_str_prefix = context_str_prefix
        self._llm = llm or llm_from_settings_or_context(Settings, service_context)
        self._text_to_sql_prompt = text_to_sql_prompt or DEFAULT_TEXT_TO_SQL_PROMPT
        self._sql_parser_mode = sql_parser_mode

        embed_model = embed_model or embed_model_from_settings_or_context(
            Settings, service_context
        )
        self._sql_parser = self._load_sql_parser(sql_parser_mode, embed_model)
        self._handle_sql_errors = handle_sql_errors
        self._sql_only = sql_only
        self._verbose = verbose
        super().__init__(
            callback_manager=callback_manager
            or callback_manager_from_settings_or_context(Settings, service_context)
        )

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "text_to_sql_prompt": self._text_to_sql_prompt,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "text_to_sql_prompt" in prompts:
            self._text_to_sql_prompt = prompts["text_to_sql_prompt"]

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt modules."""
        return {}

    def _load_sql_parser(
        self, sql_parser_mode: SQLParserMode, embed_model: BaseEmbedding
    ) -> BaseSQLParser:
        """Load SQL parser."""
        if sql_parser_mode == SQLParserMode.DEFAULT:
            return DefaultSQLParser()
        else:
            raise ValueError(f"Unknown SQL parser mode: {sql_parser_mode}")

    def _load_get_tables_fn(
        self,
        sql_database: SQLDatabase,
        tables: Optional[Union[List[str], List[Table]]] = None,
        context_query_kwargs: Optional[dict] = None,
        table_retriever: Optional[ObjectRetriever[SQLTableSchema]] = None,
    ) -> Callable[[str], List[SQLTableSchema]]:
        """Load get_tables function."""
        context_query_kwargs = context_query_kwargs or {}
        if table_retriever is not None:
            return lambda query_str: cast(Any, table_retriever).retrieve(query_str)
        else:
            if tables is not None:
                table_names: List[str] = [
                    t.name if isinstance(t, Table) else t for t in tables
                ]
            else:
                table_names = list(sql_database.get_usable_table_names())
            context_strs = [context_query_kwargs.get(t, None) for t in table_names]
            table_schemas = [
                SQLTableSchema(table_name=t, context_str=c)
                for t, c in zip(table_names, context_strs)
            ]
            return lambda _: table_schemas

    def retrieve_with_metadata(
        self, str_or_query_bundle: QueryType
    ) -> Tuple[List[NodeWithScore], Dict]:
        """Retrieve with metadata."""
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        table_desc_str = self._get_table_context(query_bundle)
        logger.info(f"> Table desc str: {table_desc_str}\n")

        response_str = self._llm.predict(
            self._text_to_sql_prompt,
            query_str=query_bundle.query_str,
            schema=table_desc_str,
            dialect=self._sql_database.dialect,
        )

        sql_query_str = self._sql_parser.parse_response_to_sql(
            response_str, query_bundle
        )
        # assume that it's a valid SQL query
        logger.info(f"> Predicted SQL query: {sql_query_str}\n")

        if self._sql_only:
            sql_only_node = TextNode(text=f"{sql_query_str}")
            retrieved_nodes = [NodeWithScore(node=sql_only_node, score=1.0)]
            metadata = {"result": sql_query_str}
        else:
            query_tables = self._get_table_from_sql(self._tables, sql_query_str)
            try:
                (
                    retrieved_nodes,
                    metadata,
                ) = self._sql_retriever.retrieve_with_metadata(sql_query_str)
                retrieved_nodes[0].metadata["invalid_flag"] = 0
                logger.info(
                    f"> SQL query result: {retrieved_nodes[0].metadata['query_output']}\n"
                )
                # if retrieved_nodes[0].metadata["query_output"] == "":

                #     new_sql_query_str = self._sql_query_modification(sql_query_str)
                #     (
                #         retrieved_nodes,
                #         metadata,
                #     ) = self._sql_retriever.retrieve_with_metadata(new_sql_query_str)
                #     logger.info(
                #         f"> Whole SQL query result: {retrieved_nodes[0].metadata['query_output']}\n"
                #     )
            except BaseException as e:
                # if handle_sql_errors is True, then return error message
                if self._handle_sql_errors:
                    logger.info(f"async error info: {e}\n")

                new_sql_query_str = self._sql_query_modification(
                    query_tables, sql_query_str
                )

                # 如果找到table，生成新的sql_query
                if new_sql_query_str != sql_query_str:
                    (
                        retrieved_nodes,
                        metadata,
                    ) = self._sql_retriever.retrieve_with_metadata(new_sql_query_str)
                    retrieved_nodes[0].metadata["invalid_flag"] = 1
                    retrieved_nodes[0].metadata[
                        "generated_query_code_instruction"
                    ] = sql_query_str
                    logger.info(
                        f"> Whole SQL query result: {retrieved_nodes[0].metadata['query_output']}\n"
                    )
                # 没有找到table，新旧sql_query一样，不再通过_sql_retriever执行，直接retrieved_nodes
                else:
                    logger.info(f"[{new_sql_query_str}] is not even a SQL")
                    retrieved_nodes = [
                        NodeWithScore(
                            node=TextNode(
                                text=new_sql_query_str,
                                metadata={
                                    "query_code_instruction": new_sql_query_str,
                                    "generated_query_code_instruction": sql_query_str,
                                    "query_output": "",
                                    "invalid_flag": 1,
                                },
                            ),
                            score=1.0,
                        ),
                    ]
                    metadata = {}

            # add query_tables into metadata
            retrieved_nodes[0].metadata["query_tables"] = query_tables

        return retrieved_nodes, {"sql_query": sql_query_str, **metadata}

    async def aretrieve_with_metadata(
        self, str_or_query_bundle: QueryType
    ) -> Tuple[List[NodeWithScore], Dict]:
        """Async retrieve with metadata."""
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle
        table_desc_str = self._get_table_context(query_bundle)
        logger.info(f"> Table desc str: {table_desc_str}\n")

        response_str = await self._llm.apredict(
            self._text_to_sql_prompt,
            query_str=query_bundle.query_str,
            schema=table_desc_str,
            dialect=self._sql_database.dialect,
        )

        sql_query_str = self._sql_parser.parse_response_to_sql(
            response_str, query_bundle
        )
        # assume that it's a valid SQL query
        logger.info(f"> Predicted SQL query: {sql_query_str}\n")

        if self._sql_only:
            sql_only_node = TextNode(text=f"{sql_query_str}")
            retrieved_nodes = [NodeWithScore(node=sql_only_node, score=1.0)]
            metadata: Dict[str, Any] = {}
        else:
            query_tables = self._get_table_from_sql(self._tables, sql_query_str)
            try:
                (
                    retrieved_nodes,
                    metadata,
                ) = await self._sql_retriever.aretrieve_with_metadata(sql_query_str)
                retrieved_nodes[0].metadata["invalid_flag"] = 0
                logger.info(
                    f"> SQL query result: {retrieved_nodes[0].metadata['query_output']}\n"
                )
                # if retrieved_nodes[0].metadata["query_output"] == "":
                #     new_sql_query_str = self._sql_query_modification(sql_query_str)
                #     (
                #         retrieved_nodes,
                #         metadata,
                #     ) = await self._sql_retriever.aretrieve_with_metadata(
                #         new_sql_query_str
                #     )
                #     logger.info(
                #         f"> Whole SQL query result: {retrieved_nodes[0].metadata['query_output']}\n"
                #     )

            except BaseException as e:
                # if handle_sql_errors is True, then return error message
                if self._handle_sql_errors:
                    logger.info(f"async error info: {e}\n")

                new_sql_query_str = self._sql_query_modification(
                    query_tables, sql_query_str
                )

                # 如果找到table，生成新的sql_query
                if new_sql_query_str != sql_query_str:
                    (
                        retrieved_nodes,
                        metadata,
                    ) = await self._sql_retriever.aretrieve_with_metadata(
                        new_sql_query_str
                    )
                    retrieved_nodes[0].metadata["invalid_flag"] = 1
                    retrieved_nodes[0].metadata[
                        "generated_query_code_instruction"
                    ] = sql_query_str
                    logger.info(
                        f"> Whole SQL query result: {retrieved_nodes[0].metadata['query_output']}\n"
                    )
                # 没有找到table，新旧sql_query一样，不再通过_sql_retriever执行，直接retrieved_nodes
                else:
                    logger.info(f"[{new_sql_query_str}] is not even a SQL")
                    retrieved_nodes = [
                        NodeWithScore(
                            node=TextNode(
                                text=new_sql_query_str,
                                metadata={
                                    "query_code_instruction": new_sql_query_str,
                                    "generated_query_code_instruction": sql_query_str,
                                    "query_output": "",
                                    "invalid_flag": 1,
                                },
                            ),
                            score=1.0,
                        ),
                    ]
                    metadata = {}

            # add query_tables into metadata
            retrieved_nodes[0].metadata["query_tables"] = query_tables

        return retrieved_nodes, {"sql_query": sql_query_str, **metadata}

    def _get_table_from_sql(self, table_list: list, sql_query: str) -> list:
        table_collection = list()
        for table in table_list:
            if table.lower() in sql_query.lower():
                table_collection.append(table)
        return table_collection

    def _sql_query_modification(self, query_tables: list, sql_query_str: str):
        # table_pattern = r"FROM\s+(\w+)"
        # match = re.search(table_pattern, sql_query_str, re.IGNORECASE | re.DOTALL)
        # if match:
        # 改用已知table匹配，否则match中FROM逻辑也可能匹配到无效的table
        if len(query_tables) != 0:
            first_table = query_tables[0]
            new_sql_query_str = f"SELECT * FROM {first_table}"
            logger.info(f"use the whole table named {first_table} instead if possible")
        else:
            # raise ValueError("No table is matched")
            new_sql_query_str = sql_query_str
            logger.info("No table is matched")

        return new_sql_query_str

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        retrieved_nodes, _ = self.retrieve_with_metadata(query_bundle)
        return retrieved_nodes

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Async retrieve nodes given query."""
        retrieved_nodes, _ = await self.aretrieve_with_metadata(query_bundle)
        return retrieved_nodes

    def _get_table_context(self, query_bundle: QueryBundle) -> str:
        """Get table context.

        Get tables schema + optional context as a single string.

        """
        table_schema_objs = self._get_tables(query_bundle.query_str)
        context_strs = []
        if self._context_str_prefix is not None:
            context_strs = [self._context_str_prefix]

        for table_schema_obj in table_schema_objs:
            table_info = self._sql_database.get_single_table_info(
                table_schema_obj.table_name
            )

            if table_schema_obj.context_str:
                table_opt_context = " The table description is: "
                table_opt_context += table_schema_obj.context_str
                table_info += table_opt_context

            context_strs.append(table_info)

        return "\n\n".join(context_strs)
