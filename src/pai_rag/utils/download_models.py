from abc import ABC, abstractmethod
from pai_rag.utils.constants import DEFAULT_MODEL_DIR, DEFAULT_MODEL_DIC
from llama_index.core.async_utils import get_asyncio_module
from concurrent.futures import ProcessPoolExecutor
from tempfile import TemporaryDirectory
from pathlib import Path
from zipfile import ZipFile
import os
import httpx
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


DEFAULT_CLIENT_TIME_OUT = 60
DEFAULT_READ_TIME_OUT = 60


class DownloadModels(ABC):
    @abstractmethod
    def load_models_from_oss(self):
        """Download models from OSS"""

        pass


class DownloadModelsFromOSS(DownloadModels):
    def __init__(self):
        self.download_directory_path = Path(DEFAULT_MODEL_DIR)
        if not os.path.exists(self.download_directory_path):
            os.makedirs(self.download_directory_path)
        self.local_model_paths = [
            os.path.join(DEFAULT_MODEL_DIR, path) for path in DEFAULT_MODEL_DIC.keys()
        ]

    async def load_models_from_oss(self, model_path):
        with TemporaryDirectory() as temp_dir:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
                model_name = model_path.split("/")[-1]
                oss_model_url = DEFAULT_MODEL_DIC.get(model_name)
                local_zip_path = os.path.join(temp_dir, model_name)

                start_time = time.time()
                timeout = httpx.Timeout(
                    DEFAULT_CLIENT_TIME_OUT, read=DEFAULT_READ_TIME_OUT
                )
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("GET", oss_model_url) as response:
                        response.raise_for_status()
                        with open(local_zip_path, "wb") as f:
                            async for chunk in response.aiter_bytes():
                                f.write(chunk)
                logger.info(f"Finished downloading mode: {local_zip_path}")
                await self.unzip_file(local_zip_path, DEFAULT_MODEL_DIR)
                end_time = time.time()
                duration = end_time - start_time
                logger.info(
                    f"Finished downloading and unzipping model: {model_path}, took {duration:.2f} seconds."
                )

    async def unzip_file(self, local_zip_path, model_path):
        executor = ProcessPoolExecutor()
        executor.submit(self._unzip_file_sync, (local_zip_path, model_path))
        # loop = asyncio.get_running_loop()
        # await loop.run_in_executor(
        #     None, self._unzip_file_sync, local_zip_path, model_path
        # )

    def _unzip_file_sync(self, local_zip_path, model_path):
        with ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(model_path)
        logger.info(f"Finished extracting {local_zip_path} to {model_path}")

    async def load_models(self, show_progress: bool = False):
        tasks = [
            self.load_models_from_oss(model_path)
            for model_path in self.local_model_paths
        ]
        _asyncio = get_asyncio_module(show_progress=show_progress)
        await asyncio.gather(*tasks)
