# Temporarily forked from file
# TODO: remove when we got a common package

import shutil
from tempfile import TemporaryDirectory
import time
from modelscope import snapshot_download
import os
from loguru import logger



INTERNAL_MODEL_DIR = "./model_repository"
PDF_EXTRACT_KIT_MODEL = "OpenDataLab/PDF-Extract-Kit-1.0"
INTERNAL_MODELS = ["BAAI/bge-m3", PDF_EXTRACT_KIT_MODEL] #必须的模型文件，提前下载


def download_model_to_directory(model_name: str):
    if model_name in INTERNAL_MODELS:
        default_model_dir = INTERNAL_MODEL_DIR
    else:
        default_model_dir = os.getenv("PAIRAG_MODEL_DIR")
    if default_model_dir:
        default_model_path = os.path.join(default_model_dir, model_name)
        if os.path.exists(default_model_path):
            logger.info(f"Model {model_name} already exists in {default_model_path}")
            return default_model_path

    model_id = model_name

    pai_model_path = os.path.join(default_model_dir, model_id)
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
