import asyncio
import click
import os
from pathlib import Path
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.modules.module_registry import module_registry
import logging

logger = logging.getLogger(__name__)


logging.basicConfig(level=logging.INFO)

_BASE_DIR = Path(__file__).parent.parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(_BASE_DIR, "config/settings.toml")


class RagDataPipeline:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    async def ingest_from_input_path(
        self,
        input_path: str,
        pattern: str,
        enable_qa_extraction: bool,
        enable_raptor: bool,
        name: str = None,
    ):
        if not name:
            # call async method will get stuck
            # maybe caused by run_in_thread wrapper we used to avoid blocking event loop
            # when uploading large files
            if isinstance(input_path, str) and os.path.isdir(input_path):
                input_paths = input_path
            else:
                input_paths = [file.strip() for file in input_path.split(",")]
            self.data_loader.load(
                input_paths,
                pattern,
                enable_qa_extraction,
                enable_raptor,
            )
        else:
            self.data_loader.load_eval_data(name)


def __init_data_pipeline(config_file, use_local_qa_model):
    config = RagConfiguration.from_file(config_file).get_value()
    module_registry.init_modules(config)

    data_loader = module_registry.get_module_with_config("DataLoaderModule", config)
    return RagDataPipeline(data_loader)


@click.command()
@click.option(
    "-c",
    "--config",
    show_default=True,
    help=f"Configuration file. Default: {DEFAULT_APPLICATION_CONFIG_FILE}",
    default=DEFAULT_APPLICATION_CONFIG_FILE,
)
@click.option(
    "-d",
    "--data-path",
    required=False,
    default=None,
    help="data path (file or directory) to ingest.",
)
@click.option(
    "-p", "--pattern", required=False, default=None, help="data pattern to ingest."
)
@click.option(
    "-q",
    "--extract-qa",
    required=False,
    is_flag=True,
    show_default=True,
    default=False,
    help="use qa metadata extraction.",
)
@click.option(
    "-l",
    "--use-local-qa-model",
    required=False,
    is_flag=True,
    show_default=True,
    default=False,
    help="use local qa extraction model.",
)
@click.option(
    "-r",
    "--enable-raptor",
    required=False,
    is_flag=True,
    show_default=True,
    default=False,
    help="use raptor for node enhancement.",
)
@click.option(
    "-n",
    "--name",
    show_default=True,
    help="Open Dataset Name. Optional: [miracl, duretrieval]",
    default=None,
)
def run(
    config,
    data_path,
    pattern,
    extract_qa,
    use_local_qa_model,
    enable_raptor,
    name,
):
    data_pipeline = __init_data_pipeline(config, use_local_qa_model)
    asyncio.run(
        data_pipeline.ingest_from_input_path(
            data_path,
            pattern,
            extract_qa,
            enable_raptor,
            name,
        )
    )
