"""retriever factory, used to generate retriever instance based on customer config and index"""

import os
import logging
from typing import Dict, List, Any

# from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.retrievers import RouterRetriever

# from llama_index.core.retrievers import NLSQLRetriever
from llama_index.core import SQLDatabase
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.indices.list.base import SummaryIndex

from pai_rag.integrations.index.multi_modal_index import MyMultiModalVectorStoreIndex
from pai_rag.integrations.retrievers.bm25 import BM25Retriever
from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.utils.prompt_template import (
    QUERY_GEN_PROMPT,
    DEFAULT_TEXT_TO_SQL_TMPL,
    DEFAULT_INSTRUCTION_STR,
    DEFAULT_PANDAS_PROMPT,
)
from pai_rag.modules.retriever.my_vector_index_retriever import MyVectorIndexRetriever
from pai_rag.integrations.retrievers.fusion_retriever import MyQueryFusionRetriever
from pai_rag.modules.retriever.my_nl2sql_retriever import MyNLSQLRetriever
from pai_rag.integrations.retrievers.data_analysis_retriever import PandasQueryRetriever

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import URL
import pandas as pd

logger = logging.getLogger(__name__)


class RetrieverModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["IndexModule", "BM25IndexModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG]
        index = new_params["IndexModule"]
        bm25_index = new_params["BM25IndexModule"]

        similarity_top_k = config.get("similarity_top_k", 5)

        retrieval_mode = config.get("retrieval_mode", "hybrid").lower()

        if retrieval_mode == "nl2sql":
            sql_database, tables, table_descriptions = self.db_connection(config)
            nl2sql_retriever = MyNLSQLRetriever(
                sql_database=sql_database,
                text_to_sql_prompt=DEFAULT_TEXT_TO_SQL_TMPL,
                tables=tables,
                context_query_kwargs=table_descriptions,
                sql_only=False,
            )
            logger.info("NL2SQLRetriever used")
            return nl2sql_retriever

        if retrieval_mode == "data_analysis":
            df = self.get_dataframe(config)
            analysis_retriever = PandasQueryRetriever(
                df=df,
                instruction_str=DEFAULT_INSTRUCTION_STR,
                pandas_prompt=DEFAULT_PANDAS_PROMPT,
            )
            logger.info("PandasQueryRetriever used")
            return analysis_retriever

        if isinstance(index.vector_index, MyMultiModalVectorStoreIndex):
            return index.vector_index.as_retriever()
        # Special handle elastic search
        elif index.vectordb_type == "milvus":
            if retrieval_mode == "embedding":
                query_mode = VectorStoreQueryMode.DEFAULT
            elif retrieval_mode == "keyword":
                query_mode = VectorStoreQueryMode.TEXT_SEARCH
            else:
                query_mode = VectorStoreQueryMode.HYBRID

            return MyVectorIndexRetriever(
                index=index.vector_index,
                similarity_top_k=similarity_top_k,
                vector_store_query_mode=query_mode,
            )
        elif index.vectordb_type == "elasticsearch":
            if retrieval_mode != "hybrid":
                if retrieval_mode == "embedding":
                    query_mode = VectorStoreQueryMode.DEFAULT
                elif retrieval_mode == "keyword":
                    query_mode = VectorStoreQueryMode.TEXT_SEARCH
                return MyVectorIndexRetriever(
                    index=index.vector_index,
                    similarity_top_k=similarity_top_k,
                    vector_store_query_mode=query_mode,
                )
            else:
                vector_retriever = MyVectorIndexRetriever(
                    index=index.vector_index,
                    similarity_top_k=similarity_top_k,
                    vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
                )
                bm25_retriever = MyVectorIndexRetriever(
                    index=index.vector_index,
                    similarity_top_k=similarity_top_k,
                    vector_store_query_mode=VectorStoreQueryMode.TEXT_SEARCH,
                )
        elif index.vectordb_type == "postgresql":
            if retrieval_mode == "embedding":
                query_mode = VectorStoreQueryMode.DEFAULT
            elif retrieval_mode == "keyword":
                query_mode = VectorStoreQueryMode.TEXT_SEARCH
            else:
                query_mode = VectorStoreQueryMode.HYBRID
            return MyVectorIndexRetriever(
                index=index.vector_index,
                similarity_top_k=similarity_top_k,
                vector_store_query_mode=query_mode,
            )
        else:
            vector_retriever = MyVectorIndexRetriever(
                index=index.vector_index, similarity_top_k=similarity_top_k
            )
            bm25_retriever = BM25Retriever.from_defaults(
                bm25_index=bm25_index,
                similarity_top_k=similarity_top_k,
            )

        if retrieval_mode == "embedding":
            logger.info(f"MyVectorIndexRetriever used with top_k {similarity_top_k}.")
            return vector_retriever

        elif retrieval_mode == "keyword":
            logger.info(f"BM25Retriever used with top_k {similarity_top_k}.")
            return bm25_retriever

        else:
            num_queries_gen = config.get("query_rewrite_n", 3)
            if config["retrieval_mode"] == "hybrid":
                fusion_retriever = MyQueryFusionRetriever(
                    [vector_retriever, bm25_retriever],
                    similarity_top_k=similarity_top_k,
                    num_queries=num_queries_gen,  # set this to 1 to disable query generation
                    use_async=True,
                    verbose=True,
                    query_gen_prompt=QUERY_GEN_PROMPT,
                )
                logger.info(f"FusionRetriever used with top_k {similarity_top_k}.")
                return fusion_retriever

            elif config["retrieval_mode"] == "router":
                nodes = list(index.vector_index.docstore.docs.values())
                summary_index = SummaryIndex(nodes)
                list_retriever = summary_index.as_retriever(
                    retriever_mode="embedding", similarity_top_k=10
                )  # can be 'default' as well for all nodes retrieval

                list_tool = RetrieverTool.from_defaults(
                    retriever=list_retriever,
                    description=("适用于总结类型的查询问题。通常伴随'总结','归纳'等关键词，一般情况下不使用。"),
                )
                vector_tool = RetrieverTool.from_defaults(
                    retriever=fusion_retriever,
                    description=("适用于一般的查询问题，首选。"),
                )

                router_retriever = RouterRetriever(
                    selector=LLMSingleSelector.from_defaults(),
                    retriever_tools=[
                        list_tool,
                        vector_tool,
                    ],
                )

                logger.info("RouterRetriever used.")
                return router_retriever

            else:
                raise ValueError("Not supported retrieval type.")

    def db_connection(self, config):
        # get rds_db config
        dialect = config.get("dialect", "sqlite")
        user = config.get("user", "")
        password = config.get("password", "")
        host = config.get("host", "")
        port = config.get("port", "")
        path = config.get("path", "")
        dbname = config.get("dbname", "")
        desired_tables = config.get("tables", [])
        table_descriptions = config.get("descriptions", {})

        if dialect == "sqlite":
            db_path = os.path.join(path, dbname)
            database_uri = f"{dialect}:///{db_path}"
        elif dialect == "mysql":
            dd_prefix = f"{dialect}+pymysql"
            database_uri = URL.create(
                dd_prefix,
                username=user,
                password=password,
                host=host,
                port=port,
                database=dbname,
            )
        else:
            raise ValueError(f"not supported SQL dialect: {dialect}")

        # use sqlalchemy engine for db connection
        engine = create_engine(database_uri, echo=False)
        inspector = inspect(engine)
        db_tables = inspector.get_table_names()
        if len(db_tables) == 0:
            raise ValueError("No database tables")

        if len(desired_tables) > 0:
            tables = desired_tables
        else:
            tables = db_tables

        # create an sqldatabase instance including desired table info
        sql_database = SQLDatabase(engine, include_tables=tables)

        if len(table_descriptions) > 0:
            table_descriptions = table_descriptions
        else:
            table_descriptions = {}

        return sql_database, tables, table_descriptions

    def get_dataframe(self, config):
        file_path = config.get("file_path", None)
        if file_path is None:
            raise ValueError("Please provide your file_path")
        _, file_extension = os.path.splitext(file_path)  # get the extension type
        if file_extension == ".csv":
            try:
                df = pd.read_csv(file_path)
                return df
            except Exception as e:
                logger.info(f"Cannot load the csv file, {e}")
            return
        elif file_extension == ".xlsx":
            try:
                df = pd.read_excel(file_path)
                return df
            except Exception as e:
                logger.info(f"Cannot load the csv file, {e}")
                return
        else:
            raise ValueError(f"Unsupported file extensions: {file_extension}\n")
