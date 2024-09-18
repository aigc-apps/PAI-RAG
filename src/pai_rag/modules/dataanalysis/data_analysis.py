import functools
import logging
import os
import glob
from typing import Dict, List, Any
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import URL
from sqlalchemy.pool import QueuePool
from llama_index.core import SQLDatabase
from llama_index.core.prompts import PromptTemplate

from pai_rag.modules.base.configurable_module import ConfigurableModule
from pai_rag.modules.base.module_constants import MODULE_PARAM_CONFIG
from pai_rag.utils.prompt_template import (
    DEFAULT_TEXT_TO_SQL_TMPL,
    DEFAULT_INSTRUCTION_STR,
    DEFAULT_PANDAS_PROMPT,
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT,
)
from pai_rag.integrations.data_analysis.data_analysis_tool import DataAnalysisTool
from pai_rag.integrations.data_analysis.nl2sql_retriever import MyNLSQLRetriever
from pai_rag.integrations.data_analysis.nl2pandas_retriever import PandasQueryRetriever
from pai_rag.integrations.data_analysis.data_analysis_synthesizer import (
    DataAnalysisSynthesizer,
)

logger = logging.getLogger(__name__)


class DataAnalysisModule(ConfigurableModule):
    @staticmethod
    def get_dependencies() -> List[str]:
        return ["LlmModule", "EmbeddingModule"]

    def _create_new_instance(self, new_params: Dict[str, Any]):
        config = new_params[MODULE_PARAM_CONFIG] or {}
        llm = new_params["LlmModule"]
        embed_model = new_params["EmbeddingModule"]
        data_analysis_type = config.get("analysis_type", "nl2pandas")
        nl2sql_prompt = config.get("nl2sql_prompt", None)
        if nl2sql_prompt:
            nl2sql_prompt = PromptTemplate(nl2sql_prompt)
        else:
            nl2sql_prompt = DEFAULT_TEXT_TO_SQL_TMPL

        if data_analysis_type == "nl2pandas":
            df = self.get_dataframe(config)
            analysis_retriever = PandasQueryRetriever(
                df=df,
                instruction_str=DEFAULT_INSTRUCTION_STR,
                pandas_prompt=DEFAULT_PANDAS_PROMPT,
                llm=llm,
            )
            logger.info("DataAnalysis PandasQueryRetriever used")

        elif data_analysis_type == "nl2sql":
            sql_database, tables, table_descriptions = self.db_connection(config)
            analysis_retriever = MyNLSQLRetriever(
                sql_database=sql_database,
                text_to_sql_prompt=nl2sql_prompt,
                tables=tables,
                context_query_kwargs=table_descriptions,
                sql_only=False,
                embed_model=embed_model,
                llm=llm,
            )
            logger.info("DataAnalysis NL2SQLRetriever used")
            # logger.info(f"nl2sql prompt: {nl2sql_prompt}")

        else:
            raise ValueError(
                "Please specify the correct analysis type, 'nl2pandas' or 'nl2sql'"
            )

        analysis_synthesizer = DataAnalysisSynthesizer(
            response_synthesis_prompt=DEFAULT_RESPONSE_SYNTHESIS_PROMPT
        )
        logger.info("DataAnalysisSynthesizer used")

        return DataAnalysisTool(
            analysis_retriever=analysis_retriever,
            analysis_synthesizer=analysis_synthesizer,
        )

    def db_connection(self, config):
        dialect = config.get("dialect", "sqlite")
        user = config.get("user", "")
        password = config.get("password", "")
        host = config.get("host", "")
        port = config.get("port", "")
        path = config.get("path", "")
        dbname = config.get("dbname", "")
        desired_tables = config.get("tables", [])
        table_descriptions = config.get("descriptions", {})

        return self.inspect_db_connection(
            dialect=dialect,
            user=user,
            password=password,
            host=host,
            port=port,
            path=path,
            dbname=dbname,
            desired_tables=tuple(desired_tables) if desired_tables else None,
            table_descriptions=tuple(table_descriptions.items())
            if table_descriptions
            else None,
        )

    @functools.cache
    def inspect_db_connection(
        self,
        dialect,
        user,
        password,
        host,
        port,
        path,
        dbname,
        desired_tables,
        table_descriptions,
    ):
        desired_tables = list(desired_tables) if desired_tables else None
        table_descriptions = dict(table_descriptions) if table_descriptions else None

        # get rds_db config
        logger.info(f"desired_tables from ui input: {desired_tables}")
        logger.info(f"table_descriptions from ui input: {table_descriptions}")

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
        engine = create_engine(
            database_uri,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=360,
            poolclass=QueuePool,
        )
        inspector = inspect(engine)
        db_tables = inspector.get_table_names()
        if len(db_tables) == 0:
            raise ValueError(f"No table found in db {dbname}.")

        if desired_tables and len(desired_tables) > 0:
            tables = desired_tables
        else:
            tables = db_tables

        # create an sqldatabase instance including desired table info
        sql_database = SQLDatabase(engine, include_tables=tables)

        if table_descriptions and len(table_descriptions) > 0:
            table_descriptions = table_descriptions
        else:
            table_descriptions = {}

        return sql_database, tables, table_descriptions

    def get_dataframe(self, config):
        file_path = config.get("file_path", "./localdata/data_analysis/")
        if not file_path:
            file_path = "./localdata/data_analysis/"

        if os.path.isfile(file_path):
            return self._read_file(file_path)
        elif os.path.isdir(file_path):
            first_file_path = self._find_first_csv_or_xlsx_in_directory(file_path)
            if first_file_path:
                return self._read_file(first_file_path)
            else:
                # raise FileExistsError("No .csv or .xlsx files found in the directory.")
                logger.info("No .csv or .xlsx files found in the directory.")
                return
        else:
            logger.info("Please provide a valid file")
            return

    def _find_first_csv_or_xlsx_in_directory(self, directory_path):
        # 使用 glob 模块查找第一个 .csv 或 .xlsx 文件
        files = glob.glob(os.path.join(directory_path, "*.csv")) + glob.glob(
            os.path.join(directory_path, "*.xlsx")
        )
        if files:
            return files[0]
        else:
            return None

    def _read_file(self, file_path):
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            return df
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
            return df
        else:
            raise TypeError("Unsupported file type.")
