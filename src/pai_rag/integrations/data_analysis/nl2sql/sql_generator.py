from loguru import logger
from typing import Any, Dict, List, Optional, Tuple
from collections import namedtuple

from llama_index.core.llms.llm import LLM
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType, TextNode
from llama_index.core import BasePromptTemplate
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import Settings
from llama_index.core.prompts.mixin import (
    PromptDictType,
    PromptMixinType,
)
from pai_rag.integrations.data_analysis.nl2sql.db_utils.nl2sql_utils import (
    MySQLRetriever,
    SQLParserMode,
    BaseSQLParser,
    DefaultSQLParser,
)
from pai_rag.integrations.data_analysis.nl2sql.db_utils.nl2sql_utils import (
    generate_schema_description,
)
from pai_rag.integrations.data_analysis.nl2sql.nl2sql_prompts import (
    DEFAULT_TEXT_TO_SQL_PROMPT,
    DEFAULT_SQL_REVISION_PROMPT,
)


ExecutionResult = namedtuple(
    "ExecutionResult", ["retrieved_nodes", "metadata", "final_sql"]
)


class SQLGenerator:
    """
    基于自然语言问题和db_info等生成候选的sql查询语句，pretriever 和 selector 非必选
    如果sql查询发生错误，结合错误信息利用llm纠错，生成revised sql
    """

    def __init__(
        self,
        sql_database: SQLDatabase,
        text_to_sql_prompt: Optional[BasePromptTemplate] = None,
        sql_revision_prompt: Optional[BasePromptTemplate] = None,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        return_raw: bool = True,
        sql_only: bool = False,
        sql_parser_mode: SQLParserMode = SQLParserMode.DEFAULT,
        **kwargs: Any,
    ) -> None:
        self._sql_database = sql_database
        self._dialect = self._sql_database.dialect
        self._tables = list(sql_database._usable_tables)
        self._sql_retriever = MySQLRetriever(sql_database, return_raw=return_raw)
        self._llm = llm or Settings.llm
        self._embed_model = embed_model or Settings.embed_model
        self._text_to_sql_prompt = text_to_sql_prompt or DEFAULT_TEXT_TO_SQL_PROMPT
        self._sql_revision_prompt = sql_revision_prompt or DEFAULT_SQL_REVISION_PROMPT
        self._sql_parser = self._load_sql_parser(sql_parser_mode, embed_model)
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

    def generate_sql_nodes(
        self,
        query_bundle: QueryType,
        selected_db_description_dict: Dict,
        selected_db_history_list: List,
        max_retry: int = 2,
    ) -> Tuple[List[NodeWithScore], Dict]:
        # step1: 获得description_str & history_str 作为llm prompt参数
        schema_description_str, _, _ = generate_schema_description(
            selected_db_description_dict
        )
        selected_db_history_str = str(selected_db_history_list)
        logger.info(f"schema_description_str for llm: {schema_description_str}")

        # step2: llm生成sql
        sql_query_str = self._agenerate_sql(
            query_bundle, schema_description_str, selected_db_history_str
        )
        logger.info(f"> Predicted SQL query: {sql_query_str}")

        ## 如果只需要返回sql语句，无需执行查询
        if self._sql_only:
            sql_only_node = TextNode(
                text=f"{sql_query_str}",
            )
            retrieved_nodes = [NodeWithScore(node=sql_only_node, score=1.0)]
            _metadata = {"result": sql_query_str}

        # step3: sql查询及纠错容错处理
        else:
            # 执行sql查询和纠错
            execute_revise_result = self._execute_and_revise(
                sql_query_str,
                max_retry,
                query_bundle,
                schema_description_str,
                selected_db_history_str,
            )

            if len(execute_revise_result.retrieved_nodes) != 0:
                retrieved_nodes = execute_revise_result.retrieved_nodes
                _metadata = execute_revise_result.metadata
            else:
                # 如果达到最大重试次数，使用朴素容错逻辑
                retrieved_nodes, _metadata = self._naive_execute_as_backup(
                    sql_query_str
                )

        return retrieved_nodes, {"sql_query": sql_query_str, **_metadata}

    async def agenerate_sql_nodes(
        self,
        query_bundle: QueryType,
        selected_db_description_dict: Dict,
        selected_db_history_list: List,
        max_retry: int = 2,
    ) -> Tuple[List[NodeWithScore], Dict]:
        # step1: 获得description_str & history_str 作为llm prompt参数
        schema_description_str, _, _ = generate_schema_description(
            selected_db_description_dict
        )
        selected_db_history_str = str(selected_db_history_list)
        logger.info(f"schema_description_str for llm: {schema_description_str}")
        logger.info(f"selected_db_history_str for llm: {selected_db_history_str}")

        # step2: llm生成sql
        sql_query_str = await self._agenerate_sql(
            query_bundle, schema_description_str, selected_db_history_str
        )
        logger.info(f"> Predicted SQL query: {sql_query_str}")

        ## 如果只需要返回sql语句，无需执行查询
        if self._sql_only:
            sql_only_node = TextNode(
                text=f"{sql_query_str}",
            )
            retrieved_nodes = [NodeWithScore(node=sql_only_node, score=1.0)]
            _metadata = {"result": sql_query_str}

        # step3: sql查询及纠错容错处理
        else:
            # 执行sql查询和纠错
            execute_revise_result = await self._aexecute_and_revise(
                sql_query_str,
                max_retry,
                query_bundle,
                schema_description_str,
                selected_db_history_str,
            )

            if len(execute_revise_result.retrieved_nodes) != 0:
                retrieved_nodes = execute_revise_result.retrieved_nodes
                _metadata = execute_revise_result.metadata
            else:
                # 如果达到最大重试次数，使用朴素容错逻辑
                retrieved_nodes, _metadata = self._naive_execute_as_backup(
                    sql_query_str
                )

        return retrieved_nodes, {"sql_query": sql_query_str, **_metadata}

    def _generate_sql(
        self,
        query_bundle: QueryBundle,
        schema_description_str: str,
        query_history_str: str,
    ) -> str:
        # llm生成response
        response_str = self._llm.predict(
            prompt=self._text_to_sql_prompt,
            dialect=self._dialect,
            query_str=query_bundle.query_str,
            db_schema=schema_description_str,
            db_history=query_history_str,
        )
        logger.info(f"> LLM response: {response_str}")

        # 解析response中的sql
        sql_query_str = self._sql_parser.parse_response_to_sql(
            response_str, query_bundle
        )

        return sql_query_str

    async def _agenerate_sql(
        self,
        query_bundle: QueryBundle,
        schema_description_str: str,
        query_history_str: str,
    ) -> str:
        # llm生成response
        response_str = await self._llm.apredict(
            prompt=self._text_to_sql_prompt,
            dialect=self._dialect,
            query_str=query_bundle.query_str,
            db_schema=schema_description_str,
            db_history=query_history_str,
        )
        logger.info(f"> LLM response: {response_str}")

        # 解析response中的sql
        sql_query_str = self._sql_parser.parse_response_to_sql(
            response_str, query_bundle
        )

        return sql_query_str

    def _execute_and_revise(
        self,
        sql_query_str: str,
        max_retry: int,
        query_bundle: QueryBundle,
        schema_description_str: str,
        db_history_str: str,
    ) -> ExecutionResult:
        attempt = 0
        last_exception = None
        while attempt < max_retry:
            attempt += 1

            try:
                execution_result = self._execute_sql_query(sql_query_str)
                logger.info(
                    f"> Attempt: {attempt}, SQL query result: {execution_result.retrieved_nodes[0]}"
                )
                return execution_result

            except NotImplementedError as error:
                last_exception = error
                logger.info(
                    f"> Attempt: {attempt}, SQL execution error: {error}, SQL query: {sql_query_str}\n"
                )
                sql_query_str = self._revise_sql(
                    error,
                    sql_query_str,
                    query_bundle,
                    schema_description_str,
                    db_history_str,
                )
                logger.info(
                    f"> Attempt: {attempt}, Revised Predicted SQL query: {sql_query_str}\n"
                )

            except Exception as error:
                last_exception = error
                logger.info(f"> Attempt: {attempt}, Unexpected error: {error}")
                raise

        logger.info(
            f"> Max retries reached, final error: {last_exception}, SQL query: {sql_query_str}\n"
        )

        return ExecutionResult([], {}, sql_query_str)

    async def _aexecute_and_revise(
        self,
        sql_query_str: str,
        max_retry: int,
        query_bundle: QueryBundle,
        schema_description_str: str,
        db_history_str: str,
    ) -> ExecutionResult:
        attempt = 0
        last_exception = None
        while attempt < max_retry:
            attempt += 1

            try:
                execution_result = self._execute_sql_query(sql_query_str)
                logger.info(
                    f"> Attempt: {attempt}, SQL query result: {execution_result.retrieved_nodes[0]}"
                )
                return execution_result

            except NotImplementedError as error:
                last_exception = error
                logger.info(
                    f"> Attempt: {attempt}, SQL execution error: {error}, SQL query: {sql_query_str}\n"
                )
                sql_query_str = await self._arevise_sql(
                    error,
                    sql_query_str,
                    query_bundle,
                    schema_description_str,
                    db_history_str,
                )
                logger.info(
                    f"> Attempt: {attempt}, Revised Predicted SQL query: {sql_query_str}\n"
                )

            except Exception as error:
                last_exception = error
                logger.info(f"> Attempt: {attempt}, Unexpected error: {error}")
                raise

        logger.info(
            f"> Max retries reached, final error: {last_exception}, SQL query: {sql_query_str}\n"
        )

        return ExecutionResult([], {}, sql_query_str)

    def _revise_sql(
        self,
        error_message: str,
        sql_query_str: str,
        nl_query_bundle: QueryBundle,
        db_schema_str: str,
        db_history_str: str,
    ) -> str:
        # 修正执行错误的SQL，同步
        response_str = self._llm.predict(
            prompt=self._sql_revision_prompt,
            dialect=self._dialect,
            query_str=nl_query_bundle.query_str,
            db_schema=db_schema_str,
            db_history=db_history_str,
            predicted_sql=sql_query_str,
            sql_execution_result=error_message,
        )
        # 解析response中的sql
        revised_sql_query = self._sql_parser.parse_response_to_sql(
            response_str, nl_query_bundle
        )
        # logger.info(f"> Revised SQL query: {revised_sql_query}\n")

        return revised_sql_query

    async def _arevise_sql(
        self,
        error_message: str,
        sql_query_str: str,
        nl_query_bundle: QueryBundle,
        db_schema_str: str,
        db_history_str: str,
    ):
        # 修正执行错误的SQL，异步
        response_str = await self._llm.apredict(
            prompt=self._sql_revision_prompt,
            dialect=self._dialect,
            query_str=nl_query_bundle.query_str,
            db_schema=db_schema_str,
            db_history=db_history_str,
            predicted_sql=sql_query_str,
            sql_execution_result=error_message,
        )
        # 解析response中的sql
        revised_sql_query = self._sql_parser.parse_response_to_sql(
            response_str, nl_query_bundle
        )
        # logger.info(f"> Async Revised SQL query: {revised_sql_query}\n")

        return revised_sql_query

    def _execute_sql_query(self, sql_query_str: str) -> ExecutionResult:
        (
            retrieved_nodes,
            _metadata,
        ) = self._sql_retriever.retrieve_with_metadata(sql_query_str)

        # 如果执行成功，标记flag，记录query_tables，并跳出循环
        retrieved_nodes[0].metadata["invalid_flag"] = 0
        query_tables = self._get_table_from_sql(self._tables, sql_query_str)
        retrieved_nodes[0].metadata["query_tables"] = query_tables

        return ExecutionResult(retrieved_nodes, _metadata, sql_query_str)

    def _get_table_from_sql(self, table_list: List[str], sql_query: str) -> List:
        """Get tables from sql query."""
        table_collection = list()
        for table in table_list:
            if table.lower() in sql_query.lower():
                table_collection.append(table)
        return table_collection

    def _naive_sql_from_table(self, query_tables: List[str], sql_query_str: str) -> str:
        """Generate naive SQL query from the first table."""
        if len(query_tables) != 0:
            first_table = query_tables[0]
            naive_sql_query_str = f"SELECT * FROM {first_table}"
            logger.info(f"Use the whole table: {first_table} instead if possible")
        else:
            naive_sql_query_str = sql_query_str
            logger.info("No table is matched")

        return naive_sql_query_str

    def _naive_execute_as_backup(
        self, sql_query_str: str
    ) -> Tuple[List[NodeWithScore], Dict]:
        # 先根据最后revised sql_query_str找到其中的tables，生成简单sql语句
        query_tables = self._get_table_from_sql(self._tables, sql_query_str)
        naive_sql_query_str = self._naive_sql_from_table(query_tables, sql_query_str)
        # 如果找到table，生成新的sql_query，执行新sql_query
        if naive_sql_query_str != sql_query_str:
            (
                retrieved_nodes,
                _metadata,
            ) = self._sql_retriever.retrieve_with_metadata(naive_sql_query_str)

            retrieved_nodes[0].metadata["invalid_flag"] = 1
            retrieved_nodes[0].metadata[
                "generated_query_code_instruction"
            ] = sql_query_str
            retrieved_nodes[0].metadata["query_tables"] = query_tables
            logger.info(
                f"> Whole SQL query result: {retrieved_nodes[0].metadata['query_output']}\n"
            )
        # 没有找到table，新旧sql_query一样，不再通过_sql_retriever执行，直接retrieved_nodes
        else:
            logger.info(f"[{sql_query_str}] failed execution")
            retrieved_nodes = [
                NodeWithScore(
                    node=TextNode(
                        text=naive_sql_query_str,
                        metadata={
                            "query_code_instruction": naive_sql_query_str,
                            "generated_query_code_instruction": sql_query_str,
                            "query_output": "",
                            "invalid_flag": 1,
                            "query_tables": "",
                        },
                    ),
                    score=1.0,
                ),
            ]
            _metadata = {}

        return retrieved_nodes, _metadata

    def _select_sql_candidates(
        self,
    ):
        """后续如需生成多个sql挑选最优时启用"""
        pass

    async def _aselect_sql_candidates(
        self,
    ):
        pass
