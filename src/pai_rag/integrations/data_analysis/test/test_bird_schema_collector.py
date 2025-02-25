from pai_rag.integrations.data_analysis.data_analysis_config import (
    SqliteAnalysisConfig,
)
from pai_rag.integrations.data_analysis.text2sql.db_connector import (
    SqliteConnector,
)
from pai_rag.integrations.data_analysis.text2sql.db_info_collector import (
    BirdSchemaCollector,
)


sqlite_config = SqliteAnalysisConfig(
    db_path="/Users/chuyu/Documents/datasets/BIRD/dev_20240627/dev_databases/california_schools/",
    database="california_schools.sqlite",
)

connector = SqliteConnector(sqlite_config)
sql_databse = connector.connect()

bird_schema_collector = BirdSchemaCollector(
    db_name="california_schools",
    sql_database=sql_databse,
    database_file_path="/Users/chuyu/Documents/datasets/BIRD/dev_20240627/dev_databases",
)

bird_schema_collector.collect()
