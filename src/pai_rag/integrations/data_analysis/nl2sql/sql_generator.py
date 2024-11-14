import logging
from typing import Any, Dict, List, Optional, Tuple

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

logger = logging.getLogger(__name__)


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
        handle_sql_errors: bool = True,
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

    def generate_sql_candidates(
        self,
        str_or_query_bundle: QueryType,
        selected_db_description_str: str,
        selected_db_history_str: str,
        max_retries: int = 1,
        candidate_num: int = 1,
    ) -> Tuple[List[NodeWithScore], Dict]:
        # 生成sql查询语句，如果candidate_num>1, 需要接candidates_selection, 暂不用
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle

        # logger.info(f"> selected_db_description_str: {selected_db_description_str}\n")

        schema_description_str, _, _ = generate_schema_description(
            selected_db_description_str
        )

        # 生成sql回答
        response_str = self._llm.predict(
            prompt=self._text_to_sql_prompt,
            dialect=self._dialect,
            query_str=query_bundle.query_str,
            db_schema=schema_description_str,
            db_history=selected_db_history_str,
        )
        logger.info(f"> LLM response: {response_str}\n")
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
            attempt = 0
            while attempt < max_retries:
                attempt += 1
                try:
                    (
                        retrieved_nodes,
                        _metadata,
                    ) = self._sql_retriever.retrieve_with_metadata(sql_query_str)
                    retrieved_nodes[0].metadata["invalid_flag"] = 0
                    logger.info(
                        f"> Attempt time: {attempt}, SQL query result: {retrieved_nodes[0].metadata['query_output']}\n"
                    )

                    # 如果执行成功，记录query_tables，并跳出循环
                    query_tables = self._get_table_from_sql(self._tables, sql_query_str)
                    break
                except Exception as e:
                    if self._handle_sql_errors:  # TODO: 直接返回错误提示，或者不需要这个功能，待定
                        logger.info(
                            f"> Attempt time: {attempt}, execution error message: {e}\n"
                        )
                    sql_execution_error_str = str(e)
                    # 调用revision尝试更新sql
                    sql_query_str = self._revise_sql(
                        nl_query_bundle=query_bundle,
                        db_schema_str=schema_description_str,
                        db_history_str=selected_db_history_str,
                        executed_sql=sql_query_str,
                        error_message=sql_execution_error_str,
                    )
                    logger.info(
                        f"> Attempt time: {attempt}, Revised Predicted SQL query: {sql_query_str}\n"
                    )

            else:
                # 如果达到最大重试次数，使用朴素容错逻辑

                # 先根据最后revised sql_query_str找到其中的tables，生成简单sql语句
                query_tables = self._get_table_from_sql(self._tables, sql_query_str)
                naive_sql_query_str = self._naive_sql_from_table(
                    query_tables, sql_query_str
                )
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
        selected_db_description_str: str,
        selected_db_history_str: str,
        max_retries: int = 1,
        candidate_num: int = 1,
    ) -> Tuple[List[NodeWithScore], Dict]:
        # 生成sql查询语句，如果candidate_num>1, 需要接candidates_selection, 暂不用
        if isinstance(str_or_query_bundle, str):
            query_bundle = QueryBundle(str_or_query_bundle)
        else:
            query_bundle = str_or_query_bundle

        # logger.info(f"> selected_db_description_str: {selected_db_description_str}\n")

        schema_description_str, _, _ = generate_schema_description(
            selected_db_description_str
        )

        # 生成sql回答
        response_str = await self._llm.apredict(
            prompt=self._text_to_sql_prompt,
            dialect=self._dialect,
            query_str=query_bundle.query_str,
            db_schema=schema_description_str,
            db_history=selected_db_history_str,
        )
        logger.info(f"> LLM response: {response_str}\n")
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
            attempt = 0
            while attempt < max_retries:
                attempt += 1
                try:
                    (
                        retrieved_nodes,
                        _metadata,
                    ) = await self._sql_retriever.aretrieve_with_metadata(sql_query_str)
                    retrieved_nodes[0].metadata["invalid_flag"] = 0
                    logger.info(
                        f"> Attempt time: {attempt}, SQL query result: {retrieved_nodes[0].metadata['query_output']}\n"
                    )
                    # 如果执行成功，记录query_tables，并跳出循环
                    query_tables = self._get_table_from_sql(self._tables, sql_query_str)
                    break
                except Exception as e:
                    if self._handle_sql_errors:  # TODO: 直接返回错误提示，或者不需要这个功能，待定
                        logger.info(
                            f"> Attempt time: {attempt}, execution error message: {e}\n"
                        )
                    sql_execution_error_str = str(e)
                    # 调用revision尝试更新sql
                    sql_query_str = await self._arevise_sql(
                        nl_query_bundle=query_bundle,
                        db_schema_str=schema_description_str,
                        db_history_str=selected_db_history_str,
                        executed_sql=sql_query_str,
                        error_message=sql_execution_error_str,
                    )
                    logger.info(
                        f"> Attempt time: {attempt}, Revised Predicted SQL query: {sql_query_str}\n"
                    )

            else:
                # 如果达到最大纠错次数，使用朴素容错逻辑

                # 先根据最后revised sql_query_str找到其中的tables，生成简单sql语句
                query_tables = self._get_table_from_sql(self._tables, sql_query_str)
                naive_sql_query_str = self._naive_sql_from_table(
                    query_tables, sql_query_str
                )
                # 如果找到table，生成新的sql_query，执行新sql_query
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
        self,
        nl_query_bundle: QueryBundle,
        db_schema_str: str,
        db_history_str: str,
        executed_sql: str,
        error_message: str,
    ) -> str:
        # 修正执行错误的SQL，同步
        response_str = self._llm.predict(
            prompt=self._sql_revision_prompt,
            dialect=self._dialect,
            query_str=nl_query_bundle.query_str,
            db_schema=db_schema_str,
            db_history=db_history_str,
            predicted_sql=executed_sql,
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
        nl_query_bundle: QueryBundle,
        db_schema_str: str,
        db_history_str: str,
        executed_sql: str,
        error_message: str,
    ):
        # 修正执行错误的SQL，异步
        response_str = await self._llm.apredict(
            prompt=self._sql_revision_prompt,
            dialect=self._dialect,
            query_str=nl_query_bundle.query_str,
            db_schema=db_schema_str,
            db_history=db_history_str,
            predicted_sql=executed_sql,
            sql_execution_result=error_message,
        )
        # 解析response中的sql
        revised_sql_query = self._sql_parser.parse_response_to_sql(
            response_str, nl_query_bundle
        )
        # logger.info(f"> Async Revised SQL query: {revised_sql_query}\n")

        return revised_sql_query
