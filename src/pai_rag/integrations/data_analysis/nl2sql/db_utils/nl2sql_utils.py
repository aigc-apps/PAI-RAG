import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import pandas as pd

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType, TextNode
from llama_index.core.instrumentation import DispatcherSpanMixin


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


def generate_schema_description(
    structured_table_description_dict: Dict,
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    基于结构化的数据库信息，生成适合llm的数据库描述，包括表结构、表描述、列描述等
    """

    if structured_table_description_dict["db_overview"]:
        schema_description_str = f"""Database overview: {structured_table_description_dict["db_overview"]}\n\n"""
    else:
        schema_description_str = ""

    table_info_list = structured_table_description_dict["table_info"]
    column_info_list = structured_table_description_dict["column_info"]
    table_info_df = pd.DataFrame(table_info_list)
    column_info_df = pd.DataFrame(column_info_list)

    # 生成所有表的描述
    all_table_descriptions = []
    for table_name in table_info_df["table"]:
        all_table_descriptions.append(
            _generate_single_table_description(
                table_name, table_info_df, column_info_df
            )
        )

    # 将所有表的描述合并成一个字符串
    schema_description_str += "\n".join(all_table_descriptions)

    return schema_description_str, table_info_df, column_info_df


def _generate_single_table_description(
    table_name, table_info_df: pd.DataFrame, column_info_df: pd.DataFrame
) -> str:
    """
    基于单表的结构化信息，生成适合llm的数据库描述，包括表结构、表描述、列描述等
    """
    if table_name not in column_info_df["table"].to_list():
        table_desc = ""
    else:
        table_row = table_info_df[table_info_df["table"] == table_name].iloc[0]
        columns = column_info_df[column_info_df["table"] == table_name]

        table_desc = f"Table {table_name} has columns: "
        for _, column in columns.iterrows():
            table_desc += f""" {column["column"]} ({column["type"]})"""
            if column["primary_key"]:
                table_desc += ", Primary Key"
            if column["foreign_key"]:
                table_desc += f""", Foreign Key, Referred Table: {column["foreign_key_referred_table"]}"""
            table_desc += f""", with Value Sample: {column["value_sample"]}"""
            if column["comment"] or column["description"]:
                table_desc += f""", with Description: {column["comment"] or ""}, {column['description'] or ""};"""
            else:
                table_desc += ";"
        if table_row["comment"] or table_row["description"] or table_row["overview"]:
            table_desc += f""" with Table Description: {table_row["comment"] or ""}, {table_row["description"] or ""}, {table_row["overview"] or ""}.\n"""
        else:
            table_desc += ".\n"

    return table_desc


# class SchemaDescription:
#     """Generate schema description str for llm"""
#     def __init__(self, db_description_dict) -> None:
#         self._db_description_dict = db_description_dict
#         self._table_info_list = db_description_dict["table_info"]
#         self._column_info_list = db_description_dict["column_info"]

#     def get_db_description_str(self) -> str:
#         """
#         整理数据库所有表的描述
#         """
#         # 获取db_overview
#         if self._db_description_dict["db_overview"]:
#             schema_description_str = f"""Database overview: {self._db_description_dict["db_overview"]}\n\n"""
#         else:
#             schema_description_str = ""
#         # 获取所有表的描述
#         all_table_descriptions = []
#         tables = [item["table"] for item in self._db_description_dict["table_info"]]
#         for table_name in tables:
#             all_table_descriptions.append(self._get_table_description_str(table_name))
#         # 拼接所有表的描述
#         schema_description_str += "\n".join(all_table_descriptions)

#         return schema_description_str

#     def _get_table_description_str(self, table_name: str) -> str:
#         """
#         根据table_name整理表的描述
#         """
#         table_column_info_list = [col for col in self._column_info_list if col["table"] == table_name]
#         table_info_dict = [table for table in self._table_info_list if table["table"] == table_name][0]

#         table_desc = f"Table {table_name} has columns: "
#         for column in table_column_info_list:
#             table_desc += f""" {column["column"]} ({column["type"]})"""
#             if column["primary_key"]:
#                 table_desc += ", primary key"
#             if column["forign_key"]:
#                 table_desc += f""", foreign key, referred table: {column["foreign_key_referred_table"]}"""
#             table_desc += f""", with value sample: {column["value_sample"]}"""
#             if column["comment"] or column["description"]:
#                 table_desc += f""", with description: {column["comment"] or ""} {column['description'] or ""};"""
#             else:
#                 table_desc += ";"
#         if table_info_dict["comment"] or table_info_dict["description"] or table_info_dict["overview"]:
#             table_desc += f""" with table description: {table_info_dict["comment"] or ""} {table_info_dict["comment"] or ""} {table_info_dict["comment"] or ""}.\n"""
#         else:
#             table_desc += ".\n"
