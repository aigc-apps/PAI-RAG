from abc import ABC
from pai_rag.utils.constants import DEFAULT_MODEL_DIR, OSS_URL
from modelscope.hub.snapshot_download import snapshot_download
from llama_index.core.async_utils import get_asyncio_module
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory
from pathlib import Path
from functools import partial
import requests
import shutil
import os
import asyncio
import time
import logging
import backoff
import click
from modelscope.hub.errors import FileIntegrityError

logger = logging.getLogger(__name__)


@backoff.on_exception(
    backoff.expo,
    FileIntegrityError,
    max_time=60,
)
class DownloadModelsFromModelScope(ABC):
    def __init__(self):
        self.download_directory_path = Path(DEFAULT_MODEL_DIR)
        if not os.path.exists(self.download_directory_path):
            os.makedirs(self.download_directory_path)
        response = requests.get(OSS_URL)
        response.raise_for_status()
        self.model_info = response.json()

    async def async_model_download(self, model_name, model_cache_dir):
        loop = asyncio.get_running_loop()
        func_to_run = partial(snapshot_download, model_name, cache_dir=model_cache_dir)
        with ThreadPoolExecutor() as pool:
            try:
                return await loop.run_in_executor(pool, func_to_run)
            except FileIntegrityError as e:
                logger.info(f"File Integrity Check Failed {e}")
                raise

    async def load_model_from_modelscope(self, model_name, model_dir):
        start_time = time.time()
        modelscope_path = self.model_info["basic_models"].get(model_name, None)
        model_path = os.path.join(model_dir, model_name)
        with TemporaryDirectory() as temp_dir:
            if not os.path.exists(model_path):
                logger.info(f"start downloading model {model_name}.")
                local_temp_path = os.path.join(temp_dir, model_name)
                os.mkdir(local_temp_path)
                temp_dir = await self.async_model_download(
                    modelscope_path, local_temp_path
                )
                local_temp_path = temp_dir
                items = os.listdir(local_temp_path)

                os.makedirs(model_path)

                for item in items:
                    source_item = os.path.join(local_temp_path, item)
                    destination_item = os.path.join(model_path, item)

                    shutil.move(source_item, destination_item)
                end_time = time.time()
                duration = end_time - start_time
                logger.info(
                    f"Finished downloading model {model_name} to {model_path}, took {duration:.2f} seconds."
                )

    async def load_basic_models(self, show_progress: bool = False):
        tasks = [
            self.load_model_from_modelscope(model_name, self.download_directory_path)
            for model_name in self.model_info["basic_models"].keys()
        ]
        _asyncio = get_asyncio_module(show_progress=show_progress)
        await asyncio.gather(*tasks)

    def load_model(self, model_name):
        model_path = os.path.join(self.download_directory_path, model_name)
        with TemporaryDirectory() as temp_dir:
            if not os.path.exists(model_path):
                start_time = time.time()
                model_id = self.model_info["extra_models"][model_name]
                temp_model_dir = snapshot_download(model_id, cache_dir=temp_dir)

                items = os.listdir(temp_model_dir)

                os.makedirs(model_path)

                for item in items:
                    source_item = os.path.join(temp_model_dir, item)
                    destination_item = os.path.join(model_path, item)

                    shutil.move(source_item, destination_item)
                end_time = time.time()
                duration = end_time - start_time
                logger.info(
                    f"Finished downloading model {model_name} to {model_path}, took {duration:.2f} seconds."
                )


@click.command()
@click.option(
    "-m",
    "--model-name",
    show_default=True,
    required=False,
    help="model name. Default: download all models provided",
    default="",
)
def load_models(model_name):
    download_models = DownloadModelsFromModelScope()
    if len(model_name) == 0:
        for model_name in download_models.model_info["extra_models"].keys():
            download_models.load_model(model_name)
    else:
        if model_name in download_models.model_info["extra_models"].keys():
            download_models.load_model(model_name)
        else:
            logger.info(f"{model_name} is not a valid model name")
