import click
import os
from pathlib import Path
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.core.rag_module import resolve_data_analysis_tool
from pai_rag.integrations.data_analysis.data_analysis_config import SqlAnalysisConfig

# from pai_rag.integrations.synthesizer.pai_synthesizer import PaiQueryBundle
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
# @click.option(
#     "-l",
#     "--input_list",
#     type=list,
#     required=True,
#     help="input list",
# )


def run(
    config_file=None,
    # input_list=None,
):
    config = RagConfigManager.from_file(config_file).get_value()
    print("config:", config)

    input_list = ["R5930 G2", "0231A5QX"]
    print("**Input List**: ", input_list)

    if isinstance(config.data_analysis, SqlAnalysisConfig):
        da = resolve_data_analysis_tool(config)

    result = da.sql_query(input_list)
    print("**Answer**: ", result)
    print([item["物料编码"] for item in result])
    print(len(result))


if __name__ == "__main__":
    run()
