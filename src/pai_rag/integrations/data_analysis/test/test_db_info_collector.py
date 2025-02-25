import os
from dotenv import load_dotenv


from pai_rag.integrations.data_analysis.data_analysis_config import (
    MysqlAnalysisConfig,
)
from pai_rag.integrations.data_analysis.text2sql.db_connector import (
    MysqlConnector,
)
from pai_rag.integrations.data_analysis.text2sql.db_info_collector import (
    SchemaCollector,
    HistoryCollector,
    ValueCollector,
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
    # tables=["pets"],
)

connector = MysqlConnector(mysql_config)
sql_database = connector.connect()
print("connector_info:", sql_database)


def test_schema_processor():
    schema_collector = SchemaCollector(
        db_name=mysql_config.database, sql_database=sql_database
    )
    schema_description = schema_collector.collect()
    assert isinstance(schema_description, dict)


def test_history_processor():
    history_collector = HistoryCollector(db_name=mysql_config.database)
    query_history = history_collector.collect()
    assert isinstance(query_history, list)


def test_value_processor():
    value_collector = ValueCollector(
        db_name=mysql_config.database, sql_database=sql_database
    )
    unique_values = value_collector.collect()
    assert isinstance(unique_values, dict)
