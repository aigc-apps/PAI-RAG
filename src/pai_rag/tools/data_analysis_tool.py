import click
import os
from pathlib import Path
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.core.rag_module import resolve_data_analysis_query

# from pai_rag.integrations.synthesizer.pai_synthesizer import PaiQueryBundle
from llama_index.core.schema import QueryBundle
import logging

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).parent.parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(
    _BASE_DIR, "config/settings_da_test.toml"
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
    logging.basicConfig(level=logging.INFO)

    config = RagConfigManager.from_file(config_file).get_value()
    print("config:", config)
    # rag_config = RagConfig.model_validate(config.rag)
    # print("rag_config:", rag_config)

    # da_loader = resolve_data_analysis_loader(rag_config)

    da_query_engine = resolve_data_analysis_query(config)

    print("**Question**: ", question)

    response_nodes = da_query_engine.retrieve(QueryBundle(query_str=question))
    print("**SQL RESULT**:", response_nodes)

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
