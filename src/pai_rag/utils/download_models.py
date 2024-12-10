from pai_rag.utils.constants import DEFAULT_MODEL_DIR, EAS_DEFAULT_MODEL_DIR, OSS_URL
from modelscope.hub.snapshot_download import snapshot_download
from tempfile import TemporaryDirectory
from pathlib import Path
import requests
import shutil
import os
import time
from loguru import logger
import click
import json


class ModelScopeDownloader:
    def __init__(self, fetch_config: bool = False, download_directory_path: str = None):
        self.download_directory_path = Path(
            download_directory_path or DEFAULT_MODEL_DIR
        )
        if fetch_config or not os.path.exists(self.download_directory_path):
            os.makedirs(self.download_directory_path, exist_ok=True)
            logger.info(
                f"Create model directory: {self.download_directory_path.resolve()} and get model info from oss {OSS_URL}."
            )
            response = requests.get(OSS_URL)
            response.raise_for_status()
            self.model_info = response.json()
            logger.info(f"Model info loaded {self.model_info}.")

    def load_model(self, model):
        model_path = self.download_directory_path / model
        if not os.path.exists(model_path):
            logger.info(
                f"Model {model} not found in {self.download_directory_path}, start downloading."
            )
            with TemporaryDirectory() as temp_dir:
                logger.info(f"start downloading model {model}.")
                start_time = time.time()
                if model in self.model_info["basic_models"]:
                    model_id = self.model_info["basic_models"][model]
                elif model in self.model_info["extra_models"]:
                    model_id = self.model_info["extra_models"][model]
                else:
                    raise ValueError(f"{model} is not a valid model name.")
                temp_model_dir = snapshot_download(model_id, cache_dir=temp_dir)
                logger.info(
                    f"Downloaded model {model} to {temp_model_dir} and start moving to {model_path}."
                )
                if not os.path.exists(model_path):
                    shutil.move(temp_model_dir, model_path)
                    end_time = time.time()
                    duration = end_time - start_time
                    logger.info(
                        f"Finished moving model {model} to {model_path.resolve()}, took {duration:.2f} seconds."
                    )
                else:
                    logger.info(
                        f"Model {model} already exists in {model_path.resolve()}, skip moving."
                    )
        else:
            logger.info(
                f"Model {model} already exists in {model_path}, skip downloading."
            )

    def load_rag_models(self, skip_download_models: bool = False):
        if not skip_download_models and DEFAULT_MODEL_DIR != EAS_DEFAULT_MODEL_DIR:
            logger.info("Not in EAS-like environment, start downloading models.")
            self.load_basic_models()
        self.load_mineru_config()

    def load_basic_models(self):
        logger.info("Start to download basic models.")
        if not hasattr(self, "model_info"):
            response = requests.get(OSS_URL)
            response.raise_for_status()
            self.model_info = response.json()
        for model in self.model_info["basic_models"].keys():
            self.load_model(model)
        logger.info("Finished downloading basic models.")

    def load_mineru_config(self):
        logger.info("Start to loading minerU config file.")
        source_path = "magic-pdf.template.json"
        destination_path = os.path.expanduser("~/magic-pdf.json")  # 目标路径

        if os.path.exists(destination_path):
            logger.info(
                "magic-pdf.json already exists, skip modifying ~/magic-pdf.json."
            )
            return

        # 读取 source_path 文件的内容
        with open(source_path, "r") as source_file:
            data = json.load(source_file)  # 加载 JSON 数据

        if "models-dir" in data:
            data["models-dir"] = (
                str(self.download_directory_path) + "/PDF-Extract-Kit/models"
            )

        # 将修改后的内容写入destination_path
        with open(destination_path, "w") as destination_file:
            json.dump(data, destination_file, indent=4)

        logger.info(
            "Copy magic-pdf.template.json to ~/magic-pdf.json and modify models-dir to model path."
        )

    def load_models(self, model=None):
        if model is None:
            models = [model for model in self.model_info["basic_models"].keys()] + [
                model for model in self.model_info["extra_models"].keys()
            ]
            for model in models:
                self.load_model(model)
        else:
            self.load_model(model)


@click.command()
@click.option(
    "-m",
    "--model-name",
    show_default=True,
    required=False,
    help="model name. Default: download all models provided",
    default=None,
)
def load_models(model_name):
    download_models = ModelScopeDownloader(fetch_config=True)
    download_models.load_models(model=model_name)
    download_models.load_mineru_config()
