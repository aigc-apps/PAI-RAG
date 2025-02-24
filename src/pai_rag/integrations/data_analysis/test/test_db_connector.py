import os
from dotenv import load_dotenv

from llama_index.core import SQLDatabase

from pai_rag.integrations.data_analysis.data_analysis_config import (
    SqliteAnalysisConfig,
    MysqlAnalysisConfig,
)
from pai_rag.integrations.data_analysis.text2sql.db_connector import (
    MysqlConnector,
    SqliteConnector,
)

# 加载 .env 文件中的环境变量
load_dotenv()

# 获取环境变量中的 API 密钥
host = os.getenv("host")
port = os.getenv("port")
user = os.getenv("user")
password = os.getenv("password")
database = os.getenv("database")

mysql_config = MysqlAnalysisConfig(
    host=host,
    port=port,
    user=user,
    password=password,
    database=database,
    tables=["pets"],
)

sqlite_config = SqliteAnalysisConfig(
    db_path="./tests/testdata/data/db_data/",
    database="pets.sqlite",
)


def test_mysql_connector():
    connector = MysqlConnector(mysql_config)
    assert connector._db_config.type == "mysql"
    sql_databse = connector.connect()
    assert isinstance(sql_databse, SQLDatabase)


def test_sqlite_connector():
    connector = SqliteConnector(sqlite_config)
    assert connector._db_config.type == "sqlite"
    sql_databse = connector.connect()
    assert isinstance(sql_databse, SQLDatabase)
