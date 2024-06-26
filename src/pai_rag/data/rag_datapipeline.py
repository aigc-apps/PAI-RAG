import asyncio
import click
import os
from pathlib import Path
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.modules.module_registry import module_registry

_BASE_DIR = Path(__file__).parent
DEFAULT_APPLICATION_CONFIG_FILE = os.path.join(_BASE_DIR, "config/settings.toml")


class RagDataPipeline:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    async def ingest_from_folder(self, folder_path: str, enable_qa_extraction: bool):
        await self.data_loader.aload(folder_path, enable_qa_extraction)


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
@click.option("-d", "--directory", required=True, help="directory path to ingest.")
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
def run(config, directory, extract_qa, use_local_qa_model):
    data_pipeline = __init_data_pipeline(config, use_local_qa_model)
    asyncio.run(data_pipeline.ingest_from_folder(directory, extract_qa))
