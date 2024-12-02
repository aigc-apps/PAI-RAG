import click
import os
import sys
from pathlib import Path
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.core.rag_module import (
    resolve_data_analysis_loader,
    resolve_data_analysis_query,
)
from pai_rag.integrations.data_analysis.data_analysis_config import SqlAnalysisConfig

# from pai_rag.integrations.synthesizer.pai_synthesizer import PaiQueryBundle
from llama_index.core.schema import QueryBundle
import logging

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent.parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(_BASE_DIR, "config/settings.toml")


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
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)]
    )

    config = RagConfigManager.from_file(config_file).get_value()
    print("config:", config)
    # rag_config = RagConfig.model_validate(config.rag)
    # print("rag_config:", rag_config)

    print("**Question**: ", question)

    if isinstance(config.data_analysis, SqlAnalysisConfig):
        da_loader = resolve_data_analysis_loader(config)
        print("check_instance:", da_loader._sql_database, id(da_loader._sql_database))
        da_loader.load_db_info()

    da_query_engine = resolve_data_analysis_query(config)
    print(
        "check_instance:",
        da_query_engine._sql_database,
        id(da_query_engine._sql_database),
    )
    # start_time1 = time.time()
    # response_nodes = da_query_engine.retrieve(QueryBundle(query_str=question))
    # end_time1 = time.time()
    # print("sql time:", round(end_time1 - start_time1, 3))

    # print("**SQL RESULT**:", response_nodes)

    if not stream:
        query_bundle = QueryBundle(query_str=question, stream=False)
        response = da_query_engine.query(query_bundle)
        print("**Answer**: ", response.response)
    else:
        query_bundle = QueryBundle(query_str=question, stream=True)
        response = da_query_engine.query(query_bundle)
        print("**Answer**: ", end="")
        for chunk in response.response_gen:
            print(chunk, end="")


if __name__ == "__main__":
    run()
