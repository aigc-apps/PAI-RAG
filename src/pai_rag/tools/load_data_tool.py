import click
import os
from pathlib import Path
from pai_rag.core.rag_config_manager import RagConfigManager
from pai_rag.utils.download_models import ModelScopeDownloader
from pai_rag.core.rag_module import resolve_data_loader
from loguru import logger


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
    "-o",
    "--oss_path",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="oss path (file or directory) to ingest. Example: oss://rag-demo/testdata",
)
@click.option(
    "-d",
    "--data_path",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="data path (file or directory) to ingest.",
)
@click.option(
    "-p",
    "--pattern",
    required=False,
    type=str,
    default=None,
    help="data pattern to ingest.",
)
@click.option(
    "-r",
    "--enable_raptor",
    required=False,
    is_flag=True,
    show_default=True,
    default=False,
    help="use raptor for node enhancement.",
)
def run(
    config_file=None,
    oss_path=None,
    data_path=None,
    pattern=None,
    enable_raptor=False,
):
    assert (oss_path is not None) or (
        data_path is not None
    ), "Must provide either local path or oss path."
    assert (oss_path is None) or (
        data_path is None
    ), f"Can not provide both local path '{data_path}' and oss path '{oss_path}'."

    config = RagConfigManager.from_file(config_file).get_value()
    ModelScopeDownloader().load_rag_models()
    data_loader = resolve_data_loader(config)
    data_loader.load_data(
        file_path_or_directory=data_path,
        filter_pattern=pattern,
        oss_path=oss_path,
        from_oss=oss_path is not None,
        enable_raptor=enable_raptor,
    )
    logger.info("Load data tool invoke finished")
