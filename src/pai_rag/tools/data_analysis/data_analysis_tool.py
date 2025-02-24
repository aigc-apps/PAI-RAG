import click
import os
from pathlib import Path
from typing import Any

from llama_index.core.schema import QueryBundle

from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.core.rag_config import RagConfig
from pai_rag.core.rag_module import resolve_llm
from pai_rag.integrations.data_analysis.data_analysis_config import SqlAnalysisConfig
from pai_rag.integrations.data_analysis.data_analysis_tool import (
    DataAnalysisConnector,
    DataAnalysisLoader,
    DataAnalysisQuery,
)

_BASE_DIR = Path(__file__).parent.parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(_BASE_DIR, "config/settings.toml")


cls_cache = {}


def resolve(cls: Any, **kwargs):
    cls_key = kwargs.__repr__()
    if cls_key not in cls_cache:
        cls_cache[cls_key] = cls(**kwargs)
    return cls_cache[cls_key]


def resolve_data_analysis_connector(config: RagConfig):
    db_connector = resolve(
        cls=DataAnalysisConnector,
        analysis_config=config.data_analysis,
    )
    return db_connector


def resolve_data_analysis_loader(config: RagConfig) -> DataAnalysisLoader:
    llm = resolve_llm(config)
    sql_database = DataAnalysisConnector(
        config.data_analysis
    ).connect()  # 每次load info都会重连数据库

    return resolve(
        cls=DataAnalysisLoader,
        analysis_config=config.data_analysis,
        sql_database=sql_database,
        llm=llm,
    )


def resolve_data_analysis_query(config: RagConfig) -> DataAnalysisQuery:
    llm = resolve_llm(config)
    sql_database = resolve_data_analysis_connector(config).connect()

    return resolve(
        cls=DataAnalysisQuery,
        analysis_config=config.data_analysis,
        sql_database=sql_database,
        llm=llm,
        callback_manager=None,
    )


@click.command()
@click.option(
    "-c",
    "--config_file",
    show_default=True,
    help=f"Configuration file. Default: {DEFAULT_APPLICATION_CONFIG_FILE}",
    default=DEFAULT_APPLICATION_CONFIG_FILE,
)
@click.option(
    "-q",
    "--question",
    type=str,
    required=True,
    help="question",
)
@click.option(
    "-s",
    "--stream",
    type=bool,
    default=False,
    required=False,
    is_flag=True,
    help="stream mode",
)
def run(
    config_file=None,
    question=None,
    stream=False,
):
    # print("config_file:", config_file)
    config = RagConfigManager.from_file(config_file).get_value()
    print("config:", config)
    # rag_config = RagConfig.model_validate(config.rag)
    # print("rag_config:", rag_config)

    print("**Question**: ", question)

    if isinstance(config.data_analysis, SqlAnalysisConfig):
        da_loader = resolve_data_analysis_loader(config)
        da_loader.load_db_info()

    da_query_engine = resolve_data_analysis_query(config)

    if not stream:
        query_bundle = QueryBundle(query_str=question)
        response = da_query_engine.query(query_bundle)
        print("**Answer**: ", response.response)
    else:
        query_bundle = QueryBundle(query_str=question)
        response = da_query_engine.query(query_bundle)
        print("**Answer**: ", end="")
        for chunk in response.response_gen:
            print(chunk, end="")


if __name__ == "__main__":
    run()
