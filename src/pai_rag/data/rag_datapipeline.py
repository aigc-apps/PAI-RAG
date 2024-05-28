import asyncio
import click
import os
from pathlib import Path
from pai_rag.data.rag_dataloader import RagDataLoader
from pai_rag.core.rag_configuration import RagConfiguration
from pai_rag.utils.oss_cache import OssCache
from pai_rag.modules.module_registry import module_registry


class RagDataPipeline:
    def __init__(self, data_loader: RagDataLoader):
        self.data_loader = data_loader

    async def ingest_from_folder(self, folder_path: str, enable_qa_extraction: bool):
        await self.data_loader.load(folder_path, enable_qa_extraction)


def __init_data_pipeline(use_local_qa_model):
    base_dir = Path(__file__).parent.parent
    config_file = os.path.join(base_dir, "config/settings.toml")

    config = RagConfiguration.from_file(config_file).get_value()
    module_registry.init_modules(config)

    oss_cache = None
    if config.get("oss_cache", None):
        oss_cache = OssCache(config.oss_cache)
    node_parser = module_registry.get_module("NodeParserModule")
    index = module_registry.get_module("IndexModule")
    data_reader_factory = module_registry.get_module("DataReaderFactoryModule")

    data_loader = RagDataLoader(
        data_reader_factory, node_parser, index, oss_cache, use_local_qa_model
    )
    return RagDataPipeline(data_loader)


@click.command()
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
def run(directory, extract_qa, use_local_qa_model):
    data_pipeline = __init_data_pipeline(use_local_qa_model)
    asyncio.run(data_pipeline.ingest_from_folder(directory, extract_qa))
