import os
import re
import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

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


def count_total_columns(db_description_dict: Dict) -> int:
    if len(db_description_dict) == 0:
        raise ValueError("db_description_dict is Empty")
    total_columns = 0
    for table in db_description_dict["table_info"]:
        total_columns += len(table["column_info"])

    return total_columns


def get_schema_desc4llm(db_description_dict: Dict) -> str:
    """get schema description for llm"""
    if len(db_description_dict) == 0:
        raise ValueError("db_description_dict is Empty")
    # 获取db_overview
    if db_description_dict["db_overview"]:
        schema_description_str = (
            f"""Database overview: {db_description_dict["db_overview"]}\n\n"""
        )
    else:
        schema_description_str = ""
    # 获取所有表的描述
    all_table_descriptions = []
    tables = [item["table_name"] for item in db_description_dict["table_info"]]
    for table_name in tables:
        all_table_descriptions.append(
            _get_table_desc(table_name, db_description_dict["table_info"])
        )
    # 拼接所有表的描述
    schema_description_str += "\n".join(all_table_descriptions)

    return schema_description_str


def _get_table_desc(table_name: str, table_info_list: List) -> str:
    """get single table description"""

    table_desc = f"Table {table_name} has columns: "
    target_table_dict = [
        table for table in table_info_list if table["table_name"] == table_name
    ][0]
    for column in target_table_dict["column_info"]:
        table_desc += f"""{column["column_name"]} ({column["column_type"]})"""
        if column["primary_key"]:
            table_desc += ", primary key"
        if column["foreign_key"]:
            table_desc += f""", foreign key, referred table: {column["foreign_key_referred_table"]}"""
        table_desc += f""", with value sample: {column["column_value_sample"]}"""
        col_comment = [
            value
            for value in [column["column_comment"], column["column_description"]]
            if value is not None
        ]
        if len(col_comment) > 0:
            table_desc += f""", with description: {", ".join(col_comment)}; """
        else:
            table_desc += "; "
    table_comment = [
        value
        for value in [
            target_table_dict["table_comment"],
            target_table_dict["table_description"],
        ]
        if value is not None
    ]
    if len(table_comment) > 0:
        table_desc += f""" with table description: {", ".join(table_comment)}."""
    else:
        table_desc += "."

    return table_desc


def get_target_info(
    target_path: str,
    target_file: Optional[Dict | List] = None,
    flag: str = "description",
) -> Dict | List:
    # 正常情况下接受传入的description dict 或 history list，否则从本地加载
    if target_file is None:
        if flag == "description":
            if not os.path.exists(target_path):
                raise ValueError(
                    f"db_description_file_path: {target_path} does not exist"
                )
        if flag == "history":
            if not os.path.exists(target_path):
                raise ValueError(f"db_history_file_path: {target_path} does not exist")
        try:
            with open(target_path, "r") as f:
                target_file = json.load(f)
        except Exception as e:
            # raise ValueError(f"Load target object from {file_path} failed: {e}")
            if flag == "description":
                target_file = {}
                logger.error(f"Error loading db_description_dict: {e}")
            if flag == "history":
                target_file = []
                logger.error(f"Error loading db_history_list: {e}")

    return target_file


def extract_subset_from_description(
    retrieved_nodes_dict: Dict, db_description_dict: Dict
) -> Dict:
    if len(retrieved_nodes_dict) > 0:
        sub_db_description_dict = {
            "db_overview": db_description_dict["db_overview"],
            "table_info": [],
        }
        for table_item in db_description_dict["table_info"]:
            filter_columns = []
            for column_item in table_item["column_info"]:
                key = (table_item["table_name"], column_item["column_name"])
                # 筛选满足条件的列并更新value sample
                if key in retrieved_nodes_dict:
                    if len(retrieved_nodes_dict[key]) > 0:
                        column_item["column_value_sample"].extend(
                            retrieved_nodes_dict[key]
                        )
                        column_item["column_value_sample"] = list(
                            set(column_item["column_value_sample"])
                        )
                    filter_columns.append(column_item)
                # 保留主键和外键
                if ((column_item["primary_key"]) or (column_item["foreign_key"])) and (
                    column_item not in filter_columns
                ):
                    filter_columns.append(column_item)
            if len(filter_columns) > 0:
                sub_db_description_dict["table_info"].append(
                    {
                        "table_name": table_item["table_name"],
                        "table_comment": table_item["table_comment"],
                        "table_description": table_item["table_description"],
                        "column_info": filter_columns,
                    }
                )
        logger.info(f"sub_db_description_dict: {sub_db_description_dict}")
        return sub_db_description_dict
    else:
        return db_description_dict
