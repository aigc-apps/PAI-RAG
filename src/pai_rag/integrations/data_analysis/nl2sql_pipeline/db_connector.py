import os
import functools
import logging
from typing import Dict
from llama_index.core import SQLDatabase
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import URL
from sqlalchemy.pool import QueuePool
from sqlalchemy import text

logger = logging.getLogger(__name__)


class DBConnector:
    """
    连接本地/远程数据库
    TODO:支持多种数据库类型, 目前支持SQLite和MySQL
    """

    def __init__(self, db_config: Dict):
        # 初始化数据库连接参数
        self.db_config = db_config
        self.engine = None

    def connect(self):
        # create sqlalchemy engine & sqldatabase instance
        dialect = self.db_config.get("dialect", "mysql")
        user = self.db_config.get("user", "")
        password = self.db_config.get("password", "")
        host = self.db_config.get("host", "")
        port = self.db_config.get("port", 3306)
        path = self.db_config.get("path", "")
        dbname = self.db_config.get("dbname", "")
        desired_tables = self.db_config.get("tables", [])
        table_descriptions = self.db_config.get("descriptions", {})

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
            # 轻量级的嵌入式数据库,整个数据库存储在一个文件中
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
        self.engine = create_engine(
            database_uri,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=360,
            poolclass=QueuePool,
        )
        inspector = inspect(self.engine)
        db_tables = inspector.get_table_names()
        if len(db_tables) == 0:
            raise ValueError(f"No table found in db {dbname}.")

        if desired_tables and len(desired_tables) > 0:
            tables = desired_tables
        else:
            tables = db_tables

        # create an sqldatabase instance including desired table info
        sql_database = SQLDatabase(self.engine, include_tables=tables)

        if table_descriptions and len(table_descriptions) > 0:
            table_descriptions = table_descriptions
        else:
            table_descriptions = {}

        return sql_database, tables, table_descriptions, dialect, dbname

    def execute_sql_query(self, sql_query: str):
        # 执行SQL查询并返回结果
        if self.engine is None:
            raise RuntimeError(
                "Database engine is not initialized. Please call connect() first."
            )
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(sql_query))
                return result.fetchall()
        except Exception as e:
            logger.info(f"An error occurred while executing the query: {e}")
            return


if __name__ == "__main__":
    config = {
        "dialect": "sqlite",
        "path": "./",
        "dbname": "test1.db",
        "desired_tables": [],
        "table_descriptions": {},
    }
    db_connector = DBConnector(db_config=config)

    # 连接数据库，创建实例
    sql_database, tables, table_descriptions, dialect, dbname = db_connector.connect()
    print(tables, dialect)

    # 执行SQL查询
    results = db_connector.execute_sql_query("SELECT * FROM pets")
    print(results)
