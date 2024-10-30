import logging
from typing import Dict, Optional

from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core import BasePromptTemplate
from llama_index.core import Settings
from llama_index.core.schema import QueryBundle

from pai_rag.integrations.data_analysis.nl2sql.db_connector import DBConnector
from pai_rag.integrations.data_analysis.nl2sql.db_descriptor import DBDescriptor
from pai_rag.integrations.data_analysis.nl2sql.nl2sql_prompts import (
    DEFAULT_DB_SUMMARY_PROMPT,
)


logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {"data_analysis": "default_setting"}  # temp mock data


class DBLoader:
    """
    offline work, including connection, description
    """

    def __init__(
        self,
        db_config: Optional[Dict] = None,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        db_summary_prompt: Optional[BasePromptTemplate] = None,
    ) -> None:
        self._db_config = db_config or DEFAULT_CONFIG
        self._dialect = self._db_config.get("dialect", "mysql")
        self._tables = self._db_config.get("tables", [])
        self._dbname = self._db_config.get("dbname", "")
        self._llm = llm or Settings.llm
        self._embed_model = embed_model or Settings.embed_model
        self._db_summary_prompt = db_summary_prompt or DEFAULT_DB_SUMMARY_PROMPT
        self._db_connector = DBConnector(db_config=self._db_config)
        logger.info("db_loader init successfully")

    def db_analysis_init(
        self,
    ):
        try:
            sql_database = self._db_connector.connect()
        except Exception as e:
            raise (
                f"cannot connect db, please check your config or network, error message: {e}"
            )

        # 2. 获得数据库描述信息
        ## 2.1 基础描述信息：ddl+sample
        db_descriptor = DBDescriptor(sql_database=sql_database, db_name=self._dbname)

        ## 2.2 数据库描述是否需要增强
        enable_enhanced_description = self._db_config.get(
            "enable_enhanced_description", True
        )
        if enable_enhanced_description is True:
            db_descriptor.get_enhanced_table_description()
        else:
            db_descriptor.get_structured_table_description(QueryBundle(""))

        ## 2.3 是否包括历史查询记录
        enable_db_history = self._db_config.get("enable_db_history", False)
        if enable_db_history is True:
            db_query_history = (
                "query_history1\nquery_history2\nquery_history3\n"  # simple mock data
            )
            db_descriptor.collect_history(db_query_history)

        ## 2.4 description embedding
        # TODO

        return sql_database
