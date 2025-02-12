from abc import ABC, abstractmethod
import os
import functools
from loguru import logger
from llama_index.core import SQLDatabase
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import URL
from sqlalchemy.pool import QueuePool

from pai_rag.integrations.data_analysis.data_analysis_config import (
    SqliteAnalysisConfig,
    MysqlAnalysisConfig,
)


# 数据库连接接口
class DBConnector(ABC):
    @abstractmethod
    def connect(self):
        pass


# mysql连接实现
class MysqlConnector(DBConnector):
    def __init__(self, db_config: MysqlAnalysisConfig):
        # 初始化数据库连接参数
        self._db_config = db_config

    def connect(self):
        """
        连接数据库
        """
        return self._inspect_db_connection(
            dialect=self._db_config.type.split(".")[0],
            user=self._db_config.user,
            password=self._db_config.password,
            host=self._db_config.host,
            port=self._db_config.port,
            dbname=self._db_config.database,
            desired_tables=tuple(self._db_config.tables)
            if self._db_config.tables
            else None,
            table_descriptions=tuple(self._db_config.descriptions.items())
            if self._db_config.descriptions
            else None,
        )

    @functools.cache
    def _inspect_db_connection(
        self,
        dialect,
        user,
        password,
        host,
        port,
        dbname,
        desired_tables,
        table_descriptions,
    ):
        desired_tables = list(desired_tables) if desired_tables else None
        table_descriptions = dict(table_descriptions) if table_descriptions else None

        # get rds_db config
        logger.info(f"desired_tables from ui input: {desired_tables}")
        logger.info(f"table_descriptions from ui input: {table_descriptions}")

        dd_prefix = f"{dialect}+pymysql"
        database_uri = URL.create(
            dd_prefix,
            username=user,
            password=password,
            host=host,
            port=port,
            database=dbname,
        )

        sql_database = _create_sqldatabase(
            database_uri, dbname, desired_tables, table_descriptions
        )

        return sql_database


# sqlite连接实现
class SqliteConnector(DBConnector):
    def __init__(self, db_config: SqliteAnalysisConfig):
        # 初始化数据库连接参数
        self._db_config = db_config

    def connect(self):
        return self._inspect_db_connection(
            dialect=self._db_config.type.split(".")[0],
            dbname=self._db_config.database,
            path=self._db_config.db_path,
            desired_tables=tuple(self._db_config.tables)
            if self._db_config.tables
            else None,
            table_descriptions=tuple(self._db_config.descriptions.items())
            if self._db_config.descriptions
            else None,
        )

    def _inspect_db_connection(
        self,
        dialect,
        dbname,
        path,
        desired_tables,
        table_descriptions,
    ):
        desired_tables = list(desired_tables) if desired_tables else None
        table_descriptions = dict(table_descriptions) if table_descriptions else None

        # get rds_db config
        logger.info(f"desired_tables from ui input: {desired_tables}")
        logger.info(f"table_descriptions from ui input: {table_descriptions}")

        if not dbname.lower().endswith(".sqlite"):
            dbname = f"{dbname}.sqlite"
        db_path = os.path.join(path, dbname)  # 轻量级的嵌入式数据库,整个数据库存储在一个文件中
        database_uri = f"{dialect}:///{db_path}"
        sql_database = _create_sqldatabase(
            database_uri, dbname, desired_tables, table_descriptions
        )

        return sql_database


def _create_sqldatabase(database_uri, dbname, desired_tables, table_descriptions):
    # use sqlalchemy engine for db connection
    engine = create_engine(
        database_uri,
        echo=False,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=360,
        poolclass=QueuePool,
    )
    inspector = inspect(engine)
    db_tables = inspector.get_table_names()
    if len(db_tables) == 0:
        raise ValueError(f"No table found in db {dbname}.")

    if desired_tables:
        valid_tables = []
        for table in desired_tables:
            if table in db_tables:
                valid_tables.append(table)
            else:
                logger.warning(f"Table {table} not found in db {dbname}.")
        if not valid_tables:
            tables = db_tables
        else:
            tables = valid_tables
    else:
        tables = db_tables

    # create an sqldatabase instance including desired table info
    sql_database = SQLDatabase(engine, include_tables=tables)

    if table_descriptions and len(table_descriptions) > 0:
        table_descriptions = table_descriptions
    else:
        table_descriptions = {}

    return sql_database
