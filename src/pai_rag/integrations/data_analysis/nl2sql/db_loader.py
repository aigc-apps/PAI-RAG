import os
from typing import Dict, Optional
from loguru import logger

from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import BasePromptTemplate
from llama_index.core import Settings
from llama_index.core.schema import QueryBundle
from llama_index.core.utilities.sql_wrapper import SQLDatabase

from pai_rag.integrations.data_analysis.nl2sql.db_descriptor import DBDescriptor
from pai_rag.integrations.data_analysis.nl2sql.db_indexer import DBIndexer
from pai_rag.integrations.data_analysis.nl2sql.nl2sql_prompts import (
    DEFAULT_DB_SUMMARY_PROMPT,
)
from pai_rag.integrations.data_analysis.data_analysis_config import (
    SqlAnalysisConfig,
    SqliteAnalysisConfig,
    MysqlAnalysisConfig,
)


class DBLoader:
    """
    offline work, including description & index
    """

    def __init__(
        self,
        db_config: Dict,
        sql_database: SQLDatabase,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        db_summary_prompt: Optional[BasePromptTemplate] = None,
    ) -> None:
        self._db_config = db_config
        self._dialect = self._db_config.get("dialect", "mysql")
        self._dbname = self._db_config.get("database", "")
        self._desired_tables = self._db_config.get("tables", [])
        self._table_descriptions = self._db_config.get("descriptions", {})
        self._sql_database = sql_database
        self._llm = llm or Settings.llm
        self._embed_model = embed_model or Settings.embed_model
        self._db_summary_prompt = db_summary_prompt or DEFAULT_DB_SUMMARY_PROMPT
        self._enable_enhanced_description = self._db_config.get(
            "enable_enhanced_description", False
        )
        self._enable_db_history = self._db_config.get("enable_db_history", False)
        self._enable_db_embedding = self._db_config.get("enable_db_embedding", False)
        self._db_descriptor = DBDescriptor(
            sql_database=self._sql_database,
            db_name=self._dbname,
            context_query_kwargs=self._table_descriptions,
            llm=self._llm,
            embed_model=self._embed_model,
        )
        self._db_indexer = DBIndexer(
            sql_database=self._sql_database,
            embed_model=self._embed_model,
        )
        logger.info("db_loader init successfully")

    def load_db_info(
        self,
    ):
        """处理数据库结构描述信息、历史查询和具体值信息, 存储json和索引"""
        # get db_description
        if self._enable_enhanced_description:
            self._llm.max_tokens = 8000
            self._db_descriptor.get_enhanced_db_description()
            logger.info("db_description obtained from with llm.")
        else:
            self._llm.max_tokens = 4000
            self._db_descriptor.get_structured_db_description(QueryBundle(""))
            logger.info("db_description obtained from without llm.")

        # get db_embedding, including db_description, db_history, db_value
        if self._enable_db_embedding:
            self._db_indexer.get_description_index()
            logger.info("db_description index stored.")

            self._db_indexer.get_value_index()
            logger.info("db_value index stored.")

            self._db_indexer.get_value_lsh()
            logger.info("db_value lsh stored.")

            if self._enable_db_history:
                self._db_indexer.get_history_index()
                logger.info("db_history index stored.")

    async def aload_db_info(
        self,
    ):
        """处理数据库结构描述信息、历史查询和具体值信息, 存储json和索引"""
        # get db_description
        if self._enable_enhanced_description:
            self._llm.max_tokens = 8000
            await self._db_descriptor.aget_enhanced_db_description()
            logger.info("db_description obtained with llm.")
        else:
            self._llm.max_tokens = 4000
            self._db_descriptor.get_structured_db_description(QueryBundle(""))
            logger.info("db_description obtained without llm.")

        # get db_embedding, including db_description, db_history, db_value
        if self._enable_db_embedding:
            await self._db_indexer.aget_description_index()
            logger.info("db_description index stored.")

            await self._db_indexer.aget_value_index()
            logger.info("db_value index stored.")

            self._db_indexer.get_value_lsh()
            logger.info("db_value lsh stored.")

            if self._enable_db_history:
                await self._db_indexer.aget_history_index()
                logger.info("db_history index stored.")

    @classmethod
    def from_config(
        cls,
        sql_config: SqlAnalysisConfig,
        sql_database: SQLDatabase,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
    ):
        db_config = {
            "dbname": sql_config.database,
            "dialect": sql_config.type.value,
            "tables": sql_config.tables,
            "descriptions": sql_config.descriptions,
            "enable_enhanced_description": sql_config.enable_enhanced_description,
            "enable_db_history": sql_config.enable_db_history,
            "enable_db_embedding": sql_config.enable_db_embedding,
        }

        if isinstance(sql_config, SqliteAnalysisConfig):
            db_path = os.path.join(sql_config.db_path, sql_config.database)
            db_config.update(
                {
                    "path": db_path,
                }
            )
        if isinstance(sql_config, MysqlAnalysisConfig):
            db_config.update(
                {
                    "user": sql_config.user,
                    "password": sql_config.password,
                    "host": sql_config.host,
                    "port": sql_config.port,
                }
            )

        return cls(
            db_config=db_config,
            sql_database=sql_database,
            llm=llm,
            embed_model=embed_model,
        )
