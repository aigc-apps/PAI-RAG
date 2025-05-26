from abc import ABC, abstractmethod
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from llama_index.core.settings import Settings
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType, TextNode
from llama_index.core.instrumentation import DispatcherSpanMixin


class DefaultSQLRetriever(BaseRetriever):
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
        super().__init__(callback_manager or Settings.callback_manager)

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

    # def _limit_check(self, sql_query: str, max_limit=100):
    #     limit_pattern = r"\bLIMIT\s+(\d+)(?:\s+OFFSET\s+\d+)?\b"
    #     match = re.search(limit_pattern, sql_query, re.IGNORECASE)

    #     if match:
    #         limit_value = int(match.group(1))
    #         if limit_value > max_limit:
    #             new_sql_query = re.sub(
    #                 limit_pattern,
    #                 f"LIMIT {max_limit}",
    #                 sql_query,
    #                 count=1,
    #                 flags=re.IGNORECASE,
    #             )
    #             return new_sql_query
    #         else:
    #             return sql_query
    #     else:
    #         raise ValueError("check sql query and regular expression")

    def _sanity_check(self, sql_query: str) -> bool:
        """
        检查给定的 SQL 查询是否包含潜在的危险字符或模式。

        参数:
            sql_query (str): 要检查的 SQL 查询字符串。

        返回:
            bool: 如果查询看起来安全，则返回 True；否则返回 False。
        """

        # 定义可能有害的模式列表
        dangerous_patterns = [
            r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE)\b"
        ]  # SQL 关键字
        # 将所有模式编译为正则表达式对象
        compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in dangerous_patterns
        ]
        # 检查每个模式是否出现在查询中
        for pattern in compiled_patterns:
            if pattern.search(sql_query):
                logger.debug(
                    f"Detected potentially dangerous pattern: {pattern.pattern}"
                )
                return False

        # 如果没有检测到任何危险模式，则认为查询是安全的
        return True

    def retrieve_with_metadata(
        self, str_or_query_bundle: QueryType
    ) -> Tuple[List[NodeWithScore], Dict]:
        """Retrieve with metadata."""
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle

        # constrain LIMIT in sql_query
        if not self._sanity_check(query_bundle.query_str):
            raise ValueError("No INSERT, UPDATE, DELETE, DROP, ALTER, CREATE in query.")
        # if "limit" not in query_bundle.query_str.lower():
        #     query_bundle.query_str = query_bundle.query_str + " limit 100"
        # else:
        #     query_bundle.query_str = self._limit_check(query_bundle.query_str)
        # logger.info(f"Limited SQL query: {query_bundle.query_str}")

        try:
            raw_response_str, metadata = self._sql_database.run_sql(
                query_bundle.query_str
            )
        except NotImplementedError as error:
            logger.info(f"Invalid SQL, error message: {error}")
            raise error

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
                        # excluded_embed_metadata_keys=[
                        #     "query_code_instruction",
                        #     "query_output",
                        #     "col_keys",
                        # ],
                        # excluded_llm_metadata_keys=[
                        #     "query_code_instruction",
                        #     "query_output",
                        #     "col_keys",
                        # ],
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
        if sql_query_start != -1:  # -1 means not found
            response = response[sql_query_start:]
            # TODO: move to removeprefix after Python 3.9+
            if response.startswith("SQLQuery:"):
                response = response[len("SQLQuery:") :]
        sql_query_end = response.find(";")
        if sql_query_end != -1:
            response = response[:sql_query_end].rstrip().replace("```", "")
        # if sql_result_start != -1:
        # response = response[:sql_result_start]
        # return response.strip().strip("```").strip().strip(";").strip().lstrip("sql")
        return response.strip().replace("```", "").lstrip("sql")
