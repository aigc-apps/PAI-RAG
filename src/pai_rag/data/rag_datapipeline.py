import asyncio
import click
import shutil
import os
import tempfile
import re
import glob
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

    async def ingest_from_files(self, file_paths: str, enable_qa_extraction: bool):
        if re.match(r"^\*\.[A-Za-z0-9_-]+$", os.path.split(file_paths)[1]):
            file_paths = glob.glob(file_paths)
        else:
            file_paths = [file.strip() for file in file_paths.split(",")]
        for file_path in file_paths:
            if not os.path.isfile(file_path):
                raise Exception(f"{file_path} is not a file path")
            try:
                temp_directory = tempfile.mkdtemp()
                save_file = os.path.join(temp_directory, os.path.basename(file_path))
                shutil.copy(file_path, save_file)

                await self.data_loader.aload(temp_directory, enable_qa_extraction)
            except Exception as e:
                raise e

    async def ingest_local_folder(self, input_path: str, enable_qa_extraction: bool):
        if not os.path.isdir(input_path):
            raise Exception(f"{input_path} is not a directory path")
        try:
            entries = os.listdir(input_path)
            paths = [os.path.join(input_path, entry) for entry in entries]

            files = [p for p in paths if os.path.isfile(p)]
            dirs = [p for p in paths if os.path.isdir(p)]

            if not dirs:
                await self.ingest_from_folder(input_path, enable_qa_extraction)

            for file_path in files:
                await self.ingest_from_files(file_path, enable_qa_extraction)

            for dir_path in dirs:
                await self.ingest_local_folder(dir_path, enable_qa_extraction)
        except Exception as e:
            raise e


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
    "-t",
    "--type",
    type=click.Choice(["d", "f"]),
    required=True,
    help="type: d for directory, f for files",
)
@click.option("-p", "--paths", required=True, help="directory or files path to ingest.")
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
def run(config, type, paths, extract_qa, use_local_qa_model):
    data_pipeline = __init_data_pipeline(config, use_local_qa_model)
    if type == "d":
        asyncio.run(data_pipeline.ingest_local_folder(paths, extract_qa))
    elif type == "f":
        asyncio.run(data_pipeline.ingest_from_files(paths, extract_qa))
    else:
        raise click.BadParameter("need to choose d (directory) or f (files)")
