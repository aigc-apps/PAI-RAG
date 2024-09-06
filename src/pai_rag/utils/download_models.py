from pai_rag.utils.constants import DEFAULT_MODEL_DIR, OSS_URL
from modelscope.hub.snapshot_download import snapshot_download
from tempfile import TemporaryDirectory
from pathlib import Path
import requests
import shutil
import os
import time
import logging
import click
import json

logger = logging.getLogger(__name__)


class ModelScopeDownloader:
    def __init__(self):
        self.download_directory_path = Path(DEFAULT_MODEL_DIR)
        if not os.path.exists(self.download_directory_path):
            os.makedirs(self.download_directory_path)
        response = requests.get(OSS_URL)
        response.raise_for_status()
        self.model_info = response.json()

    def load_model(self, model_name):
        model_path = os.path.join(self.download_directory_path, model_name)
        with TemporaryDirectory() as temp_dir:
            if not os.path.exists(model_path):
                logger.info(f"start downloading model {model_name}.")
                start_time = time.time()
                if model_name in self.model_info["basic_models"]:
                    model_id = self.model_info["basic_models"][model_name]
                elif model_name in self.model_info["extra_models"]:
                    model_id = self.model_info["extra_models"][model_name]
                else:
                    raise ValueError(f"{model_name} is not a valid model name.")
                temp_model_dir = snapshot_download(model_id, cache_dir=temp_dir)

                shutil.move(temp_model_dir, model_path)
                end_time = time.time()
                duration = end_time - start_time
                logger.info(
                    f"Finished downloading model {model_name} to {model_path}, took {duration:.2f} seconds."
                )

    def load_basic_models(self):
        for model_name in self.model_info["basic_models"].keys():
            self.load_model(model_name)

    def load_mineru_config(self):
        source_path = "magic-pdf.template.json"
        destination_path = os.path.expanduser("~/magic-pdf.json")  # 目标路径

        if os.path.exists(destination_path):
            print("magic-pdf.json already exists, skip modifying ~/magic-pdf.json.")
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

        print(
            "Copy magic-pdf.template.json to ~/magic-pdf.json and modify models-dir to model path."
        )

    def load_models(self, model_name):
        if model_name is None:
            model_names = [
                model_name for model_name in self.model_info["basic_models"].keys()
            ] + [model_name for model_name in self.model_info["extra_models"].keys()]
            for model_name in model_names:
                self.load_model(model_name)
        else:
            self.load_model(model_name)


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
    download_models = ModelScopeDownloader()
    download_models.load_models(model_name)
    download_models.load_mineru_config()
