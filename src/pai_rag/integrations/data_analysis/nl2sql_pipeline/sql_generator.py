import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from sqlalchemy import Table

from llama_index.core.llms.llm import LLM
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType, TextNode
from llama_index.core import PromptTemplate, BasePromptTemplate
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import Settings
from llama_index.core.prompts.mixin import (
    PromptDictType,
    PromptMixinType,
)
from pai_rag.integrations.data_analysis.nl2sql_pipeline.nl2sql_utils import (
    MySQLRetriever,
    SQLParserMode,
    BaseSQLParser,
    DefaultSQLParser,
)

logger = logging.getLogger(__name__)


DEFAULT_TEXT_TO_SQL_PROMPT = PromptTemplate(
    "给定一个输入问题，创建一个语法正确的{dialect}查询语句来执行。\n"
    "不要从特定的表中查询所有列, 只根据问题查询几个相关的列。\n"
    "请注意只使用你在schema descriptions 和 query history 中看到的列名。\n"
    "不要查询不存在的列。\n"
    "请注意哪个列位于哪个表中。必要时，请使用表名限定列名。\n\n"
    "Question: {query_str} \n"
    "Schema descriptions: {schema} \n"
    "Query history: {db_history} \n"
    "You are required to use the following format, each taking one line:\n\n"
    "Question: Question here\n"
    "SQLQuery: SQL Query to run \n\n"
)

DEFAULT_SQL_REVISION_PROMPT = PromptTemplate(
    "Given an input question, database schema, sql execution result and query history, revise the predicted sql query following the correct {dialect} based on the instructions below.\n"
    "Instructions:\n"
    "1. When you need to find the highest or lowest values based on a certain condition, using ORDER BY + LIMIT 1 is preferred over using MAX/MIN within sub queries.\n"
    "2. If predicted query includes an ORDER BY clause to sort the results, you should only include the column(s) used for sorting in the SELECT clause if the question specifically ask for them. Otherwise, omit these columns from the SELECT.\n"
    "3. If the question doesn't specify exactly which columns to select, between name column and id column, prefer to select id column.\n"
    "4. Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.\n"
    "5. Predicted query should return all of the information asked in the question without any missing or extra information.\n"
    "6. No matter of how many things the question asks, you should only return one SQL query as the answer having all the information asked in the question, separated by a comma.\n"
    "7. When ORDER BY is used, just include the column name in the ORDER BY in the SELECT clause when explicitly asked in the question. Otherwise, do not include the column name in the SELECT clause.\n\n"
    "Question: {query_str}\n"
    "Database schema: {schema}\n"
    "Query history: {db_history}\n"
    "Predicted sql query: {predicted_sql}\n"
    "SQL execution result: {sql_execution_result}\n"
    "You are required to use the following format, each taking one line:\n\n"
    "Question: Question here\n"
    "SQLQuery: SQL Query to run \n\n"
)


class SQLGenerator:
    """
    基于自然语言问题和db_info等生成候选的sql查询语句，pretriever 和 selector 非必选
    如果sql查询发生错误，结合错误信息利用llm纠错，生成revised sql
    """

    def __init__(
        self,
        dialect: str,
        sql_database: SQLDatabase,
        text_to_sql_prompt: Optional[BasePromptTemplate] = None,
        sql_revision_prompt: Optional[BasePromptTemplate] = None,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        tables: Optional[Union[List[str], List[Table]]] = None,
        db_schema: str = "",
        db_history: str = "",
        return_raw: bool = True,
        sql_only: bool = False,
        sql_parser_mode: SQLParserMode = SQLParserMode.DEFAULT,
        handle_sql_errors: bool = False,
        **kwargs: Any,
    ) -> None:
        self._dialect = dialect
        self._sql_database = sql_database
        self._tables = tables
        self._sql_retriever = MySQLRetriever(sql_database, return_raw=return_raw)
        self._text_to_sql_prompt = text_to_sql_prompt or DEFAULT_TEXT_TO_SQL_PROMPT
        self._sql_revision_prompt = sql_revision_prompt or DEFAULT_SQL_REVISION_PROMPT
        self._llm = llm or Settings.llm
        self._embed_model = embed_model or Settings.embed_model
        self._db_schema = db_schema
        self._db_history = db_history
        self._sql_parser = self._load_sql_parser(sql_parser_mode, embed_model)
        self._handle_sql_errors = handle_sql_errors
        self._sql_only = sql_only

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {
            "text_to_sql_prompt": self._text_to_sql_prompt,
            "sql_revision_prompt": self._sql_revision_prompt,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "text_to_sql_prompt" in prompts:
            self._text_to_sql_prompt = prompts["text_to_sql_prompt"]
        if "sql_revision_prompt" in prompts:
            self._sql_revision_prompt = prompts["sql_revision_prompt"]

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

    def _get_table_from_sql(self, table_list: List[str], sql_query: str) -> list:
        table_collection = list()
        for table in table_list:
            if table.lower() in sql_query.lower():
                table_collection.append(table)
        return table_collection

    def _naive_sql_from_table(self, query_tables: List[str], sql_query_str: str) -> str:
        if len(query_tables) != 0:
            first_table = query_tables[0]
            naive_sql_query_str = f"SELECT * FROM {first_table}"
            logger.info(f"use the whole table named {first_table} instead if possible")
        else:
            naive_sql_query_str = sql_query_str
            logger.info("No table is matched")

        return naive_sql_query_str

    def generate_sql_candidates(
        self,
        str_or_query_bundle: QueryType,
        candidate_num: int = 1,
    ) -> Tuple[List[NodeWithScore], Dict]:
        # 生成sql查询语句，如果candidate_num>1, 需要接candidates_selection, 暂不用
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle

        logger.info(f"> de_schema_desc: {self._db_schema}\n")

        # 生成sql回答
        response_str = self._llm.predict(
            prompt=self._text_to_sql_prompt,
            query_str=query_bundle.query_str,
            schema=self._db_schema,
            dialect=self._dialect,
            db_history=self._db_history,
        )
        # 解析回答中的sql
        sql_query_str = self._sql_parser.parse_response_to_sql(
            response_str, query_bundle
        )
        logger.info(f"> Predicted SQL query: {sql_query_str}\n")

        # 如果只需要返回sql语句，无需执行查询
        if self._sql_only:
            sql_only_node = TextNode(
                text=f"{sql_query_str}",
            )
            retrieved_nodes = [NodeWithScore(node=sql_only_node, score=1.0)]
            _metadata = {"result": sql_query_str}

        # 执行sql查询
        else:
            # 执行第一次查询，如出现查询错误，调用revise_sql纠正
            try:
                (
                    retrieved_nodes,
                    _metadata,
                ) = self._sql_retriever.retrieve_with_metadata(sql_query_str)
                retrieved_nodes[0].metadata["invalid_flag"] = 0
                logger.info(
                    f"> 1st SQL query result: {retrieved_nodes[0].metadata['query_output']}\n"
                )
            except Exception as e:
                if self._handle_sql_errors:  # TODO: 直接返回错误提示，或者不需要这个功能，待定
                    logger.info(f"> 1st execution error message: {e}\n")
                sql_execution_error = str(e)
                sql_query_str = self._revise_sql(
                    nl_query_bundle=query_bundle,
                    executed_sql=sql_query_str,
                    error_message=sql_execution_error,
                )
                logger.info(f"> Revised Predicted SQL query: {sql_query_str}\n")

            # 匹配sql_query_str中的tables
            query_tables = self._get_table_from_sql(self._tables, sql_query_str)

            # 执行第二次查询，如出现错误，使用简单容错逻辑
            try:
                (
                    retrieved_nodes,
                    _metadata,
                ) = self._sql_retriever.retrieve_with_metadata(sql_query_str)
                retrieved_nodes[0].metadata["invalid_flag"] = 0
                logger.info(
                    f"> 2nd SQL query result: {retrieved_nodes[0].metadata['query_output']}\n"
                )
            except Exception as e:
                # if handle_sql_errors is True, then return error message
                if self._handle_sql_errors:
                    logger.info(f"> 2nd execution error message: {e}\n")

                naive_sql_query_str = self._naive_sql_from_table(
                    query_tables, sql_query_str
                )
                # 如果找到table，生成新的sql_query
                if naive_sql_query_str != sql_query_str:
                    (
                        retrieved_nodes,
                        _metadata,
                    ) = self._sql_retriever.retrieve_with_metadata(naive_sql_query_str)

                    retrieved_nodes[0].metadata["invalid_flag"] = 1
                    retrieved_nodes[0].metadata[
                        "generated_query_code_instruction"
                    ] = sql_query_str
                    logger.info(
                        f"> Whole SQL query result: {retrieved_nodes[0].metadata['query_output']}\n"
                    )
                # 没有找到table，新旧sql_query一样，不再通过_sql_retriever执行，直接retrieved_nodes
                else:
                    logger.info(f"[{sql_query_str}] is not even a SQL")
                    retrieved_nodes = [
                        NodeWithScore(
                            node=TextNode(
                                text=naive_sql_query_str,
                                metadata={
                                    "query_code_instruction": naive_sql_query_str,
                                    "generated_query_code_instruction": sql_query_str,
                                    "query_output": "",
                                    "invalid_flag": 1,
                                },
                            ),
                            score=1.0,
                        ),
                    ]
                    _metadata = {}

            # add query_tables into metadata
            retrieved_nodes[0].metadata["query_tables"] = query_tables

        return retrieved_nodes, {"sql_query": sql_query_str, **_metadata}

    async def agenerate_sql_candidates(
        self,
        str_or_query_bundle: QueryType,
        candidate_num: int = 1,
    ) -> Tuple[List[NodeWithScore], Dict]:
        # 生成sql查询语句，如果n>1, 需要接candidates_selection, 暂不用
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle

        logger.info(f"> de_schema_desc: {self._db_schema}\n")

        # 生成sql回答
        response_str = await self._llm.apredict(
            prompt=self._text_to_sql_prompt,
            query_str=query_bundle.query_str,
            schema=self._db_schema,
            dialect=self._dialect,
            db_history=self._db_history,
        )
        # 解析回答中的sql
        sql_query_str = self._sql_parser.parse_response_to_sql(
            response_str, query_bundle
        )
        logger.info(f"> Predicted SQL query: {sql_query_str}\n")

        # 如果只需要返回sql语句，无需执行查询
        if self._sql_only:
            sql_only_node = TextNode(
                text=f"{sql_query_str}",
            )
            retrieved_nodes = [NodeWithScore(node=sql_only_node, score=1.0)]
            _metadata = {"result": sql_query_str}

        # 执行sql查询
        else:
            # 执行第一次查询，如出现查询错误，调用revise_sql纠正
            try:
                (
                    retrieved_nodes,
                    _metadata,
                ) = await self._sql_retriever.aretrieve_with_metadata(sql_query_str)
                retrieved_nodes[0].metadata["invalid_flag"] = 0
                logger.info(
                    f"> 1st SQL query result: {retrieved_nodes[0].metadata['query_output']}\n"
                )
            except Exception as e:
                if self._handle_sql_errors:  # TODO: 直接返回错误提示，或者不需要这个功能，待定
                    logger.info(f"> 1st execution error message: {e}\n")
                sql_execution_error = str(e)
                sql_query_str = await self._arevise_sql(
                    nl_query_bundle=query_bundle,
                    executed_sql=sql_query_str,
                    error_message=sql_execution_error,
                )
                logger.info(f"> Revised Predicted SQL query: {sql_query_str}\n")

            # 匹配sql_query_str中的tables
            query_tables = self._get_table_from_sql(self._tables, sql_query_str)

            # 执行第二次查询，如出现错误，使用简单容错逻辑
            try:
                (
                    retrieved_nodes,
                    _metadata,
                ) = await self._sql_retriever.aretrieve_with_metadata(sql_query_str)
                retrieved_nodes[0].metadata["invalid_flag"] = 0
                logger.info(
                    f"> 2nd SQL query result: {retrieved_nodes[0].metadata['query_output']}\n"
                )
            except Exception as e:
                # if handle_sql_errors is True, then return error message
                if self._handle_sql_errors:
                    logger.info(f"> 2nd execution error message: {e}\n")

                naive_sql_query_str = self._naive_sql_from_table(
                    query_tables, sql_query_str
                )
                # 如果找到table，生成新的sql_query
                if naive_sql_query_str != sql_query_str:
                    (
                        retrieved_nodes,
                        _metadata,
                    ) = await self._sql_retriever.aretrieve_with_metadata(
                        naive_sql_query_str
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
                    logger.info(f"[{sql_query_str}] is not even a SQL")
                    retrieved_nodes = [
                        NodeWithScore(
                            node=TextNode(
                                text=naive_sql_query_str,
                                metadata={
                                    "query_code_instruction": naive_sql_query_str,
                                    "generated_query_code_instruction": sql_query_str,
                                    "query_output": "",
                                    "invalid_flag": 1,
                                },
                            ),
                            score=1.0,
                        ),
                    ]
                    _metadata = {}

            # add query_tables into metadata
            retrieved_nodes[0].metadata["query_tables"] = query_tables

        return retrieved_nodes, {"sql_query": sql_query_str, **_metadata}

    def _select_sql_candidates(
        self,
    ):
        pass

    async def _aselect_sql_candidates(
        self,
    ):
        pass

    def _revise_sql(
        self, nl_query_bundle: QueryBundle, executed_sql: str, error_message: str
    ):
        # 修正执行错误的SQL，同步
        response_str = self._llm.predict(
            prompt=self._sql_revision_prompt,
            query_str=nl_query_bundle.query_str,
            schema=self._db_schema,
            db_history=self._db_history,
            dialect=self._dialect,
            predicted_sql=executed_sql,
            sql_execution_result=error_message,
        )
        # 解析response中的sql
        revised_sql_query = self._sql_parser.parse_response_to_sql(
            response_str, nl_query_bundle
        )
        logger.info(f"> Revised SQL query: {revised_sql_query}\n")

        return revised_sql_query

    async def _arevise_sql(
        self, nl_query_bundle: QueryBundle, executed_sql: str, error_message: str
    ):
        # 修正执行错误的SQL，异步
        response_str = await self._llm.apredict(
            prompt=self._sql_revision_prompt,
            query_str=nl_query_bundle.query_str,
            schema=self._db_schema,
            db_history=self._db_history,
            dialect=self._dialect,
            predicted_sql=executed_sql,
            sql_execution_result=error_message,
        )
        # 解析response中的sql
        revised_sql_query = self._sql_parser.parse_response_to_sql(
            response_str, nl_query_bundle
        )
        logger.info(f"> Async Revised SQL query: {revised_sql_query}\n")

        return revised_sql_query

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        retrieved_nodes, _ = self.generate_sql_candidates(query_bundle)
        return retrieved_nodes

    async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Async retrieve nodes given query."""
        retrieved_nodes, _ = await self.agenerate_sql_candidates(query_bundle)
        return retrieved_nodes
