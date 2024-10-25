import click
import os
from pathlib import Path
from pai_rag.core.rag_config import RagConfig
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.core.rag_module import resolve_query_engine
from pai_rag.integrations.synthesizer.pai_synthesizer import PaiQueryBundle
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
    logging.basicConfig(level=logging.INFO)

    config = RagConfigManager.from_file(config_file).get_value()
    rag_config = RagConfig.model_validate(config.rag)

    query_engine = resolve_query_engine(rag_config)

    print("**Question**: ", question)

    if not stream:
        query_bundle = PaiQueryBundle(query_str=question, stream=False)
        response = query_engine.query(query_bundle)
        print("**Answer**: ", response.response)
    else:
        query_bundle = PaiQueryBundle(query_str=question, stream=True)
        response = query_engine.query(query_bundle)
        print("**Answer**: ", end="")
        for chunk in response.response_gen:
            print(chunk, end="")


if __name__ == "__main__":
    run()
