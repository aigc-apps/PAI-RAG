# Temporarily forked from pairag.file
# TODO: remove when we got a common package

import shutil
from tempfile import TemporaryDirectory
import time
from modelscope import snapshot_download
from pairag.file.readers.pai.utils.cuda_utils import infer_cuda_device
from pathlib import Path
import os
import json
from loguru import logger

DEFAULT_MODEL_DIR = "./localdata/model_repository"

modelscope_id_map = {
    "PDF-Extract-Kit-1.0": "Ceceliachenen/PDF-Extract-Kit-1.0",
    "bge-m3": "BAAI/bge-m3",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "bge-reranker-large": "BAAI/bge-reranker-large",
}


def init_mineru_config(model_dir: str = DEFAULT_MODEL_DIR):
    # 获取配置文件目录
    download_model_to_directory("PDF-Extract-Kit-1.0", model_dir=model_dir)

    current_dir_path = Path(__file__).parent.parent.parent.parent
    source_path = os.path.join(current_dir_path, "magic-pdf.template.json")

    logger.info(f"Start to loading minerU config file from {source_path}.")
    destination_path = os.path.expanduser("~/magic-pdf.json")  # 目标路径

    # 读取 source_path 文件的内容
    with open(source_path, "r") as source_file:
        data = json.load(source_file)  # 加载 JSON 数据

    data["device-mode"] = infer_cuda_device()

    if "models-dir" in data:
        data["models-dir"] = os.path.join(str(model_dir), "PDF-Extract-Kit-1.0/models")
    if "layoutreader-model-dir" in data:
        data["layoutreader-model-dir"] = os.path.join(
            str(model_dir),
            "PDF-Extract-Kit-1.0/models/layoutreader",
        )

    # 将修改后的内容写入destination_path
    with open(destination_path, "w") as destination_file:
        json.dump(data, destination_file, indent=4)

    logger.info(
        f"Copy {source_path} to ~/magic-pdf.json and modify models-dir to model path."
    )


def download_model_to_directory(model_name: str, model_dir: str = DEFAULT_MODEL_DIR):
    default_model_dir = os.getenv("PAIRAG_MODEL_DIR")  # 在EAS上不可写入
    if default_model_dir:
        default_model_path = os.path.join(default_model_dir, model_name)
        if os.path.exists(default_model_path):
            logger.info(f"Model {model_name} already exists in {default_model_path}")
            return default_model_path

    if model_name in modelscope_id_map:
        model_id = modelscope_id_map[model_name]
    else:
        model_id = model_name

    pai_model_path = os.path.join(model_dir, model_id)
    logger.info(f"Model {model_id} not found, start downloading to {pai_model_path}.")

    if not os.path.exists(pai_model_path):
        with TemporaryDirectory() as temp_dir:
            start_time = time.time()
            logger.info(f"start downloading model {model_id}.")
            temp_model_dir = snapshot_download(model_id, cache_dir=temp_dir)
            logger.info(
                f"Downloaded model {model_id} to {temp_model_dir} and start moving to {pai_model_path}."
            )
            if not os.path.exists(pai_model_path):
                shutil.move(temp_model_dir, pai_model_path)
                end_time = time.time()
                duration = end_time - start_time
                logger.info(
                    f"Finished moving model {model_name} to {pai_model_path}, took {duration:.2f} seconds."
                )
            else:
                logger.info(
                    f"Model {model_name} already exists in {pai_model_path}, skip moving."
                )
    else:
        logger.info(f"Model {model_name} already downloaded.")

    return pai_model_path
