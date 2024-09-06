import os
import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import URL

from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core import Settings
from llama_index.core import SQLDatabase

from pai_rag.utils.prompt_template import DEFAULT_TEXT_TO_SQL_TMPL
from pai_rag.integrations.data_analysis.nl2sql_retriever import (
    MyNLSQLRetriever,
    MySQLRetriever,
)

load_dotenv()

llm = DashScope(model_name="qwen-max", temperature=0.1)
embed_model = DashScopeEmbedding(embed_batch_size=10)
Settings.llm = llm
Settings.embed_model = embed_model


@pytest.fixture()
def db_connection():
    if os.path.exists("./env"):
        dialect = os.getenv("dialect")
        user = os.getenv("user")
        password = os.getenv("password")
        host = os.getenv("host")
        port = os.getenv("port")
        path = os.getenv("path")
        dbname = os.getenv("dbname")
        desired_tables = os.getenv("tables")
        table_descriptions = os.getenv("descriptions")
    else:
        dialect = "sqlite"
        path = "./tests/testdata/data/db_data"
        dbname = "pets.db"
        desired_tables = []
        table_descriptions = {}

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
        raise ValueError(f"No table found in db {dbname}.")

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


@pytest.mark.skipif(
    os.getenv("DASHSCOPE_API_KEY") is None, reason="no llm api key provided"
)
def test_sql_retriever(db_connection):
    sql_database, db_tables, table_descriptions = db_connection
    sql_retriever = MySQLRetriever(sql_database=sql_database)
    sql_query = "SELECT * FROM student"

    res = sql_retriever.retrieve(sql_query)

    assert res[0].metadata["query_code_instruction"] == sql_query + " limit 100"


@pytest.mark.skipif(
    os.getenv("DASHSCOPE_API_KEY") is None, reason="no llm api key provided"
)
def test_nl2sql_retriever(db_connection):
    sql_database, db_tables, table_descriptions = db_connection
    nl2sql_retriever = MyNLSQLRetriever(
        sql_database=sql_database,
        text_to_sql_prompt=DEFAULT_TEXT_TO_SQL_TMPL,
        tables=db_tables,
    )

    res = nl2sql_retriever.retrieve("找出体重大于10的宠物的数量")

    assert res[0].score == 1
